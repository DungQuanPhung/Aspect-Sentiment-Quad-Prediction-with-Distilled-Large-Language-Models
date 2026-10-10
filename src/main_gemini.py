import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import *
import models
import ast
import json
import re
import argparse
from collections import defaultdict
import tqdm
import time

def parse_file(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
        for line in lines:
            sentence, quadruples_str = line.strip().split('####')
            quadruples = ast.literal_eval(quadruples_str)
            quadruples = [quad[:4] for quad in quadruples]
            data.append({
                'sentence': sentence,
                'quadruples': quadruples
            })
    return data

def category_polarity_extraction(data):
    out = list()
    for i in data:
        category_polarity_tuples = list()
        for q in i["quadruples"]:
            category_polarity_tuples.append((q[1], q[2]))
        out.append({"sentence": i['sentence'], "category_polarity": category_polarity_tuples})
    return out

def save_as_json(data, output_file):
    with open(output_file, 'w', encoding='utf-8') as file:
        json.dump(data, file, indent=4, ensure_ascii=False)

def read_previous_json(inference_filename):
    """Counts already-predicted instances so a run resumes where the previous one stopped."""
    processed_dir = os.path.join(data_path, "processed")
    # exact `<prefix>_<n>.json` match, otherwise e.g. `hotels` would also pick up `hotels_vn` files
    pattern = re.compile(rf"^{re.escape(inference_filename)}_(\d+)\.json$")
    json_files = [file for file in os.listdir(processed_dir) if pattern.match(file)]
    count_instances = 0
    for json_file in json_files:
        with open(os.path.join(processed_dir, json_file), 'r', encoding='utf-8') as f:
            json_data = json.load(f)
            count_instances += len(json_data[list(json_data.keys())[0]])
    return count_instances, len(json_files)

def save_timing(inference_filename, **info):
    timing_dir = os.path.join(data_path, "processed", "timing")
    os.makedirs(timing_dir, exist_ok=True)
    info["sec_per_instance"] = info["total_sec"] / info["n_instances"] if info["n_instances"] else 0
    with open(os.path.join(timing_dir, inference_filename.replace(".json", "_timing.json")), 'w',
              encoding='utf-8') as file:
        json.dump(info, file, indent=4)

def select_instances(data, instances, limit):
    return data[instances:] if limit is None else data[instances:instances + limit]

def cot_chainer(*, input_file, ground_truth_file, dataset_name, domain, model, run_name=None, limit=None):
    run_name = run_name or dataset_name
    data = parse_file(input_file)
    save_as_json(data, ground_truth_file)
    print(f"Data has been saved to {ground_truth_file}")
    inference_filename = re.sub(r"[/:]", "_", model) + "_agents_" + run_name
    instances, file_n = read_previous_json(inference_filename)
    inference_filename = inference_filename + "_" + str(file_n + 1) + ".json"
    logging.basicConfig(filename=os.path.join(data_path, "processed", "log",
                                              ".".join(inference_filename.split('.')[:-1]) + ".log"),
                        level=logging.DEBUG,
                        format='%(asctime)s - %(levelname)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        encoding='utf-8')
    logger = logging.getLogger(__name__)
    Categories.set_categories(dataset_name)
    fill_icl_template_for_cot_agents(dataset_name)
    agents = defaultdict(list)
    todo = select_instances(data, instances, limit)
    print(f"{instances} instances already done, running {len(todo)} more")
    start, calls_before = time.perf_counter(), models.n_calls
    done = 0
    try:
        for i in tqdm.tqdm(todo):
            print(i['sentence'])
            logger.info(i['sentence'])
            for quads, agent_id in chainer(i['sentence'], domain=domain, model=model):
                agents[agent_id].append(list(quads))
                logger.info(quads[1])
            done += 1
            with open(os.path.join(data_path, "processed", inference_filename), 'w', encoding='utf-8') as file:
                json.dump(agents, file, indent=4, ensure_ascii=False)
    finally:
        # written even on interruption so partial runs still have their timing
        if done:
            save_timing(inference_filename, model=model, task="agents", dataset=run_name, n_instances=done,
                        total_sec=time.perf_counter() - start, n_api_calls=models.n_calls - calls_before)

def icl(*, input_file, ground_truth_file, dataset_name, model, examples, run_name=None, limit=None):
    run_name = run_name or dataset_name
    data = parse_file(input_file)
    save_as_json(data, ground_truth_file)
    print(f"Data has been saved to {ground_truth_file}")
    inference_filename = re.sub(r"[/:]", "_", model) + "_icl_" + str(examples) + "_" + run_name
    instances, file_n = read_previous_json(inference_filename)
    inference_filename = inference_filename + "_" + str(file_n + 1) + ".json"
    print(inference_filename)
    icl_template_ = fill_icl_template_for_icl(dataset_name, number_of_examples=examples)
    system_instruction = """You are a Natural Language Processing assistant, expert in Aspect-Based Sentiment Analysis. 
    Follow the instructions and do what you have been asked without explanations or reasoning."""
    agents = defaultdict(list)
    agent_key = "agent_icl_" + str(examples)
    todo = select_instances(data, instances, limit)
    print(f"{instances} instances already done, running {len(todo)} more")
    start, calls_before = time.perf_counter(), models.n_calls
    try:
        for i in tqdm.tqdm(todo):
            out = execute_prompt(chat_history=icl_template_, prompt=i['sentence'],
                                 system_instruction=system_instruction,
                                 model=model)
            agents[agent_key].append(out)
    finally:
        done = len(agents[agent_key])
        if done:
            with open(os.path.join(data_path, "processed", inference_filename), 'w', encoding='utf-8') as file:
                json.dump(agents, file, indent=4, ensure_ascii=False)
            save_timing(inference_filename, model=model, task="icl", examples=examples, dataset=run_name,
                        n_instances=done, total_sec=time.perf_counter() - start,
                        n_api_calls=models.n_calls - calls_before)

def preliminary_prompt_experiments(*, input_file, ground_truth_file, dataset_name, model):
    data = parse_file(input_file)
    data = category_polarity_extraction(data)
    print(dataset_name)
    return preliminary_prompt_experiment(data, model)

def task_switch(task, dataset_name, model_path, split="test", limit=None, **kwargs):
    os.makedirs(os.path.join(data_path, "processed", "log"), exist_ok=True)
    input_file = os.path.join(data_path, "raw", dataset_name, split + ".txt")
    # non-test splits get their own output names so they never mix with the test predictions
    run_name = dataset_name if split == "test" else f"{dataset_name}_{split}"
    ground_truth_file = os.path.join(data_path, "processed", run_name + "_ground_truth.json")
    if task == 'cot_chainer':
        cot_chainer(input_file=input_file, ground_truth_file=ground_truth_file, dataset_name=dataset_name,
                    model=model_path, domain=kwargs['domain'], run_name=run_name, limit=limit)
    elif task == 'icl':
        icl(input_file=input_file, ground_truth_file=ground_truth_file, dataset_name=dataset_name, model=model_path,
            examples=kwargs['examples'], run_name=run_name, limit=limit)
    elif task == 'preliminary_prompt_experiments':
        preliminary_prompt_experiments(input_file=input_file, ground_truth_file=ground_truth_file,
                                       dataset_name=dataset_name, model=model_path)

DOMAINS = {"rest15": "restaurant reviews", "rest16": "restaurant reviews", "laptop_acos": "laptop reviews",
           "amazon_FF": "amazon fine foods reviews", "hotels": "hotel reviews", "shoes": "shoes reviews",
           "hotels_vn": "hotel reviews", "hotels_vn_demo": "hotel reviews"}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gemini-2.0-flash-001")
    parser.add_argument("--datasets", nargs="+", default=["hotels_vn"], choices=sorted(DOMAINS))
    parser.add_argument("--tasks", nargs="+", default=["cot_chainer", "icl"], choices=["cot_chainer", "icl"])
    parser.add_argument("--examples", nargs="+", type=int, default=[0, 2, 10, 20, 50, 100],
                        help="numbers of in-context examples to try for the icl task")
    parser.add_argument("--split", default="test", choices=["train", "dev", "test"])
    parser.add_argument("--limit", type=int, default=None, help="only run the next N unprocessed instances")
    parser.add_argument("--hf-4bit", action="store_true", help="load hf: models in 4-bit (needs bitsandbytes)")
    args = parser.parse_args()
    models.LOCAL_4BIT = args.hf_4bit
    print("Starting the timer...")
    start_time = time.time()
    for dataset_name in args.datasets:
        if "cot_chainer" in args.tasks:
            task_switch(task="cot_chainer", dataset_name=dataset_name, model_path=args.model,
                        domain=DOMAINS[dataset_name], split=args.split, limit=args.limit)
        if "icl" in args.tasks:
            for example in args.examples:
                task_switch(task="icl", dataset_name=dataset_name, model_path=args.model, examples=example,
                            split=args.split, limit=args.limit)
    end_time = time.time()
    print(f"Time taken for the code block: {end_time - start_time} seconds")