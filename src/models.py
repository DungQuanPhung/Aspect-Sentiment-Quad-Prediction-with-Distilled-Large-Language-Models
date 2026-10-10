import os
import concurrent.futures
import re
import time
import warnings
import toml
import pprint
import transformers
import torch

# Cloud SDKs are optional so that local `hf:` models run without them installed.
try:
    import vertexai
    import vertexai.preview.generative_models as generative_models
    from google.api_core import exceptions as google_exceptions
    from vertexai.generative_models import GenerativeModel
    from vertexai.generative_models._generative_models import Content, Part
except ImportError:
    vertexai = None
try:
    from openai import OpenAI, RateLimitError, APIConnectionError, InternalServerError
except ImportError:
    OpenAI = None
try:
    from anthropic import AnthropicVertex
except ImportError:
    AnthropicVertex = None

pipeline = None
pipeline_model_id = None
n_calls = 0  # total LLM requests made in this process, used for cost/timing reports

LOCAL_PREFIX = "hf:"  # e.g. "hf:Qwen/Qwen2.5-3B-Instruct" runs the model in-process with transformers
LOCAL_4BIT = False  # load the local model with bitsandbytes nf4 (set via --hf-4bit)

models = ['gpt-4o', 'gpt-4', 'gpt-4-turbo', 'gemini-1.5-pro-001', 'gemini-1.5-flash-001', 'gemini-1.5-flash-002', 'gemini-1.0-pro-002', 'gpt-4o-2024-08-06',
          'claude-3-opus@20240229', 'claude-3-sonnet@20240229', 'claude-3-haiku@20240307', 'Qwen/Qwen2.5-3B-Instruct', 'mlx-community/Qwen2.5-3B-Instruct-bf16',
          'Qwen/Qwen3-4B', 'Qwen/Qwen2.5-3B-Instruct', 'gemini-2.5-flash-preview-04-17', 'gemini-2.0-flash-001', 'Qwen/Qwen3-30B-A3B']

config_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'config')

prev_model = None
api_key_params, parameters = None, None

class TimeoutException(Exception):
    """Custom exception for function timeouts."""
    pass


def run_with_timeout(func, chat_history, prompt, timeout):
    """Runs a function with a timeout.

    Thread-based so it also works on Windows, where signal.SIGALRM is unavailable.

    Args:
        func (function): The function to run.
        chat_history (list): The chat history to pass to the function.
        prompt (str): The prompt to pass to the function.
        timeout (int): The timeout in seconds.

    Returns:
        The result of the function.
    """
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    future = executor.submit(func, chat_history, prompt)
    try:
        return future.result(timeout=timeout)
    except concurrent.futures.TimeoutError:
        raise TimeoutException("Function execution exceeded time limit")
    finally:
        executor.shutdown(wait=False)


def _local_pipeline(model_id):
    global pipeline, pipeline_model_id
    if pipeline is None or pipeline_model_id != model_id:
        dtype_key = "dtype" if int(transformers.__version__.split(".")[0]) >= 5 else "torch_dtype"
        model_kwargs = {}
        if LOCAL_4BIT:
            model_kwargs["quantization_config"] = transformers.BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16)
        elif torch.cuda.is_available():
            model_kwargs[dtype_key] = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        else:
            model_kwargs[dtype_key] = torch.float32
        pipeline = transformers.pipeline(task="text-generation", model=model_id, model_kwargs=model_kwargs,
                                         device_map="auto")
        pipeline_model_id = model_id
    return pipeline


class _StopAfterQuads(transformers.StoppingCriteria):
    """Stops once the quad list is closed, so a model that never emits EOS cannot run to max_new_tokens."""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, input_ids, scores, **kwargs):
        tail = self.tokenizer.decode(input_ids[0, -3:], skip_special_tokens=True)
        return torch.full((input_ids.shape[0],), "]]" in tail, dtype=torch.bool, device=input_ids.device)


def _run_local(model_id, messages):
    pipe = _local_pipeline(model_id)
    outputs = pipe(messages, max_new_tokens=1024, do_sample=False, return_full_text=False,
                   stopping_criteria=transformers.StoppingCriteriaList([_StopAfterQuads(pipe.tokenizer)]))
    text = outputs[0]["generated_text"]
    if isinstance(text, list):  # older transformers return the whole chat instead of only the new text
        text = text[-1]["content"]
    # reasoning models (Qwen3, ...) prepend a <think> block that is not part of the answer
    return re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()

def execute_prompt(*, chat_history, prompt, system_instruction, model, vertex_ai_max_attempts=10):
    """Executes a prompt using the specified model.

    This function provides a unified interface for interacting with various LLMs,
    including those from OpenAI (GPT), Google (Gemini), Anthropic (Claude), and local Qwen models.
    It handles API-specific authentication, request formatting, and error handling.

    Args:
        chat_history (list): The history of the conversation.
        prompt (str): The user's prompt.
        system_instruction (str): The system instruction for the model.
        model (str): The name of the model to use.
        vertex_ai_max_attempts (int, optional): The maximum number of attempts for Vertex AI models. Defaults to 10.

    Returns:
        str: The model's response.
    """
    global n_calls

    def read_params(model_):
        """Reads the parameters for the specified model."""
        global prev_model, api_key_params, parameters
        if prev_model != model_:
            if not model_.startswith(LOCAL_PREFIX):
                with open(os.path.join(config_dir, 'api_keys.toml'), 'r') as params:
                    api_key_params = toml.load(params)
                    os.environ['OPENAI_API_KEY'] = api_key_params['open_ai']['api_key']
                    if vertexai is not None:
                        vertexai.init(project=api_key_params['vertex_ai']['project'],
                                      location=api_key_params['vertex_ai']['location'])
            parameters = None
            try:
                with open(os.path.join(config_dir, model_ + '.toml'), 'r') as params:
                    parameters = toml.load(params)
            except OSError:  # no config for this model (or unusable file name such as `hf:org/model`)
                pass
            pprint.pprint(parameters)
        prev_model = model_

    read_params(model)
    n_calls += 1

    if model.startswith(LOCAL_PREFIX):
        _chat_history = list()
        _chat_history.append({"role": "system", "content": system_instruction})
        for ch in chat_history:
            if 'content' not in ch:
                ch['content'] = ch['text']
            _chat_history.append({"role": ch["role"], "content": ch["content"]})
        _chat_history.append({"role": "user", "content": prompt})
        return _run_local(model[len(LOCAL_PREFIX):], _chat_history)

    if model not in models:
        raise Exception("Invalid model. Choose from {}".format(models))
    if ("gpt" in model or "Qwen" in model) and OpenAI is None:
        raise ImportError("The `openai` package is required for GPT and vLLM-served Qwen models")
    if ("claude" in model and AnthropicVertex is None) or ("gemini" in model and vertexai is None):
        raise ImportError("`anthropic` / `google-cloud-aiplatform` are required for Claude and Gemini models")
    if "gpt" in model:
        client = OpenAI()
        _chat_history = list()
        _chat_history.append({"role": "system", "content": system_instruction})
        for ch in chat_history:
            if 'content' not in ch:
                ch['content'] = ch['text']
            _chat_history.append({"role": ch["role"], "content": ch["content"]})
        _chat_history.append({"role": "user", "content": prompt})
        while True:
            try:
                response = client.chat.completions.create(
                    max_tokens=1024,
                    model=model,
                    temperature=0,
                    messages=[*_chat_history]
                )
                break
            except (RateLimitError, APIConnectionError, InternalServerError) as e:
                print(e)
                time.sleep(20)
        res_dump = response.model_dump()
        return res_dump['choices'][0]['message']['content']

    elif "Qwen" in model:
        openai_api_base = "http://localhost:8000/v1"
        openai_api_key = "token-abc123"

        client = OpenAI(
            api_key=openai_api_key,
            base_url=openai_api_base,
        )
        _chat_history = list()
        _chat_history.append({"role": "system", "content": system_instruction})
        for ch in chat_history:
            if 'content' not in ch:
                ch['content'] = ch['text']
            _chat_history.append({"role": ch["role"], "content": ch["content"]})
        _chat_history.append({"role": "user", "content": prompt})
        while True:
            try:
                response = client.chat.completions.create(
                    max_tokens=1024,
                    model=model,
                    temperature=0,
                    messages=[*_chat_history],
                    extra_body={
                        "chat_template_kwargs": {"enable_thinking": False},
                    },
                )
                break
            except (RateLimitError, APIConnectionError, InternalServerError) as e:
                print(e)
                time.sleep(20)
        res_dump = response.model_dump()
        return res_dump['choices'][0]['message']['content']

    elif "claude" in model:
        client = AnthropicVertex(region=api_key_params['vertex_ai']['location'],
                                 project_id=api_key_params['vertex_ai']['project'])

        _chat_history = list()
        for ch in chat_history:
            if 'content' not in ch:
                ch['content'] = ch['text']
            _chat_history.append({"role": ch["role"], "content": ch["content"]})
        _chat_history.append({"role": "user", "content": prompt})

        message = client.messages.create(
            max_tokens=512,
            messages=[{"role": "user", "content": "Hello, Claude!"}],
            model=model,
        )
        return message

    elif "gemini" in model:
        def run(chat_history_, prompt_):
            global parameters
            nonlocal safety_settings
            chat = model.start_chat(history=chat_history_, response_validation=False)
            responses_ = chat.send_message(prompt_, stream=False, safety_settings=safety_settings,
                                           generation_config=parameters['parameters'])
            return responses_

        model = GenerativeModel(model_name=model,
                                system_instruction=system_instruction)
        safety_settings = {
            generative_models.HarmCategory.HARM_CATEGORY_HATE_SPEECH: generative_models.HarmBlockThreshold.BLOCK_NONE,
            generative_models.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: generative_models.HarmBlockThreshold.BLOCK_NONE,
            generative_models.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: generative_models.HarmBlockThreshold.BLOCK_NONE,
            generative_models.HarmCategory.HARM_CATEGORY_HARASSMENT: generative_models.HarmBlockThreshold.BLOCK_NONE}

        for attempt in range(vertex_ai_max_attempts):
            _chat_history = list()
            for ch in chat_history:
                if ch['role'] == "assistant":
                    ch['role'] = 'model'
                if 'content' not in ch:
                    ch['content'] = ch['text']
                _chat_history.append(Content(role=ch['role'], parts=[Part.from_text(ch['content'])]))
            try:
                assert len(_chat_history) == len(chat_history)
            except AssertionError as e:
                print(e)
            try:
                responses = run_with_timeout(run, chat_history=_chat_history, prompt=prompt, timeout=30)
            except google_exceptions.InvalidArgument as e:
                print(e)
                break
            except TimeoutException as e:
                print(e)
                print("waiting...")
                time.sleep(243)
                continue
            except (google_exceptions.ResourceExhausted, google_exceptions.FailedPrecondition, IndexError,
                    google_exceptions.InternalServerError) as e:
                print(e)
                print("waiting...")
                time.sleep(312)
                continue
            res = responses.to_dict()['candidates'][0]
            if res['finish_reason'] == 'MAX_TOKENS':
                print(responses.text)
                return "[]"
            elif res['finish_reason'] == 'OTHER':
                time.sleep(25)
                continue
            break
        else:
            warnings.warn("Exceeded times of attempts")
            return "[]"
        try:
            return responses.text
        except ValueError:
            raise Exception(responses.text)

    raise Exception(f"No backend for model {model!r}; prefix HuggingFace models with {LOCAL_PREFIX!r}")

if __name__ == '__main__':
    system_instruction = "You count the apples and print an integer. Only that, please."
    ch_1 = {'role': 'user', 'text': 'You know that George has one apple.'}
    ch_2 = {'role': 'model', 'text': 'George has one apple.'}
    ch_3 = {'role': 'user', 'text': 'You know that Mary has two apples.'}

    chat_history = [ch_1, ch_2]

    r = execute_prompt(model="gemini-1.5-flash-001", chat_history=chat_history, system_instruction=system_instruction,
                       prompt=ch_3['text'])
    print(r)