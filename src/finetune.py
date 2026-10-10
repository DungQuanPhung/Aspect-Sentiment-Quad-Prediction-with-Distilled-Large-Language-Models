"""LoRA / QLoRA fine-tuning of a HuggingFace chat model for ASQP on a data/raw/<dataset> split.

After training, the best adapter (by dev quad-F1) predicts the test split and the files are written in the same
layout as main_*.py, so `python src/eval.py --model <repo> --dataset <name> --tasks ft` scores them; the timing
file additionally carries the real training time. Run from the repository root.
"""
import argparse
import ast
import inspect
import json
import math
import os
import random
import re
import sys
import time

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
parser.add_argument("--dataset", default="hotels_vn")
parser.add_argument("--epochs", type=float, default=15)
parser.add_argument("--patience", type=int, default=3,
                    help="early stopping: stop when dev quad-F1 has not improved for this many epochs (0 = off)")
parser.add_argument("--lr", type=float, default=2e-4)
parser.add_argument("--batch-size", type=int, default=4)
parser.add_argument("--grad-accum", type=int, default=4)
parser.add_argument("--max-len", type=int, default=768, help="training examples longer than this (tokens) are skipped")
parser.add_argument("--lora-r", type=int, default=16)
parser.add_argument("--load-4bit", action="store_true", help="QLoRA: 4-bit base model (needed for 7B on 16 GB)")
parser.add_argument("--dtype", default="auto", choices=["auto", "fp16", "bf16", "fp32"],
                    help="base model precision; auto = bf16 when the GPU supports it, else fp16 (use fp32 if loss is nan)")
parser.add_argument("--dev-samples", type=int, default=300,
                    help="dev rows scored after every epoch for model selection / early stopping (0 = keep last epoch)")
parser.add_argument("--max-train-samples", type=int, default=None, help="subsample the training split (smoke tests)")
parser.add_argument("--gen-batch-size", type=int, default=16)
parser.add_argument("--max-new-tokens", type=int, default=384)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--predict-only", metavar="ADAPTER_DIR", help="skip training; predict the test split with this adapter")
parser.add_argument("--all-gpus", action="store_true", help="let Trainer use every visible GPU (default: GPU 0 only)")
args = parser.parse_args()
if not args.all_gpus:
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")  # must be set before torch is imported

import torch
import transformers
from peft import (LoraConfig, PeftModel, get_peft_model, get_peft_model_state_dict, prepare_model_for_kbit_training,
                  set_peft_model_state_dict)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import eval as asqp_eval  # src/eval.py: the repo's quad-F1 scorer

data_path = os.path.join(os.getcwd(), "data")
# Short instruction: a fine-tuned model learns the label set, so the long ICL prompt would only cost tokens.
SYSTEM_PROMPT = ("You are an Aspect-Based Sentiment Analysis expert. Extract every aspect sentiment quadruple from the "
                 "review as a Python list of [aspect, category, polarity, opinion]. Use 'NULL' for implicit aspects or "
                 "opinions. Output only the list.")
LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def read_split(split):
    rows = []
    with open(os.path.join(data_path, "raw", args.dataset, f"{split}.txt"), encoding="utf-8") as f:
        for line in f:
            sentence, quads = line.rstrip("\n").split("####")
            rows.append({"sentence": sentence, "quadruples": [q[:4] for q in ast.literal_eval(quads)]})
    return rows


def messages_for(sentence):
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": sentence}]


def encode_example(tokenizer, row):
    """Prompt + answer token ids, with the prompt masked out of the loss."""
    prompt = tokenizer.apply_chat_template(messages_for(row["sentence"]), tokenize=False, add_generation_prompt=True)
    full = tokenizer.apply_chat_template(
        messages_for(row["sentence"]) + [{"role": "assistant", "content": repr(row["quadruples"])}], tokenize=False)
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    full_ids = tokenizer(full, add_special_tokens=False)["input_ids"]
    if len(full_ids) > args.max_len:
        return None
    return {"input_ids": full_ids, "labels": [-100] * len(prompt_ids) + full_ids[len(prompt_ids):]}


def collate(batch, pad_id):
    width = max(len(b["input_ids"]) for b in batch)
    ids = torch.full((len(batch), width), pad_id)
    labels = torch.full((len(batch), width), -100)
    mask = torch.zeros((len(batch), width), dtype=torch.long)
    for i, b in enumerate(batch):
        n = len(b["input_ids"])
        ids[i, :n] = torch.tensor(b["input_ids"])
        labels[i, :n] = torch.tensor(b["labels"])
        mask[i, :n] = 1
    return {"input_ids": ids, "attention_mask": mask, "labels": labels}


def load_base(model_id):
    if args.dtype == "auto":
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16  # T4/P100 have no bf16
    else:
        dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[args.dtype]
    dtype_key = "dtype" if int(transformers.__version__.split(".")[0]) >= 5 else "torch_dtype"
    kwargs = {dtype_key: dtype, "device_map": {"": 0}}
    if args.load_4bit:
        kwargs["quantization_config"] = transformers.BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=dtype)
    return transformers.AutoModelForCausalLM.from_pretrained(model_id, **kwargs), dtype


@torch.no_grad()
def generate(model, tokenizer, sentences):
    was_training = model.training
    model.eval()
    tokenizer.padding_side = "left"
    outputs = []
    for i in range(0, len(sentences), args.gen_batch_size):
        prompts = [tokenizer.apply_chat_template(messages_for(s), tokenize=False, add_generation_prompt=True)
                   for s in sentences[i:i + args.gen_batch_size]]
        enc = tokenizer(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
        gen = model.generate(**enc, max_new_tokens=args.max_new_tokens, do_sample=False, use_cache=True,
                             pad_token_id=tokenizer.pad_token_id)
        outputs.extend(tokenizer.batch_decode(gen[:, enc["input_ids"].shape[1]:], skip_special_tokens=True))
    tokenizer.padding_side = "right"
    if was_training:
        model.train()
    return [o.strip() for o in outputs]


def quad_f1(rows, predictions):
    tp = fp = fn = 0
    for row, pred in zip(rows, predictions):
        try:
            quads = asqp_eval.string_parsing(pred)
        except Exception:
            quads = []
        a, b, c = asqp_eval.calculate_tp_fp_fn_asqp(row["quadruples"], quads)
        tp, fp, fn = tp + a, fp + b, fn + c
    return asqp_eval.calculate_fscore(tp, fp, fn)


class DevQuadF1(transformers.TrainerCallback):
    """Scores a dev sample after every epoch, keeps the best adapter and stops early once dev quad-F1 stalls."""

    def __init__(self, tokenizer, dev_rows, out_dir, patience):
        self.tokenizer, self.dev_rows, self.out_dir, self.patience = tokenizer, dev_rows, out_dir, patience
        self.best_f1, self.best_state, self.history, self.bad_epochs, self.stopped_early = -1.0, None, [], 0, False

    def on_epoch_end(self, training_args, state, control, model=None, **kwargs):
        preds = generate(model, self.tokenizer, [r["sentence"] for r in self.dev_rows])
        p, r, f1 = quad_f1(self.dev_rows, preds)
        self.history.append({"epoch": round(state.epoch, 2), "dev_precision": p, "dev_recall": r, "dev_quad_f1": f1})
        print(f"\n[epoch {state.epoch:.0f}] dev quad-F1 {f1:.4f} (P {p:.4f} R {r:.4f})", flush=True)
        if f1 > self.best_f1:
            self.best_f1, self.bad_epochs = f1, 0
            self.best_state = {k: v.detach().cpu().clone() for k, v in get_peft_model_state_dict(model).items()}
            model.save_pretrained(os.path.join(self.out_dir, "best"))
        else:
            self.bad_epochs += 1
            if self.patience and self.bad_epochs >= self.patience:
                print(f"early stopping: no dev improvement for {self.patience} epochs", flush=True)
                self.stopped_early = True
                control.should_training_stop = True
        return control


def next_run_index(prefix):
    processed = os.path.join(data_path, "processed")
    pattern = re.compile(rf"^{re.escape(prefix)}_(\d+)\.json$")
    return max((int(m.group(1)) for f in os.listdir(processed) if (m := pattern.match(f))), default=0) + 1


def main():
    assert torch.cuda.is_available(), "a CUDA GPU is required"
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    tag = re.sub(r"[/:]", "_", args.model)
    out_dir = os.path.join(data_path, "processed", "finetune", f"{tag}_{args.dataset}")
    os.makedirs(out_dir, exist_ok=True)
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model, dtype = load_base(args.model)

    train_info = {}
    if args.predict_only:
        model = PeftModel.from_pretrained(model, args.predict_only)
    else:
        train_rows = read_split("train")
        if args.max_train_samples:
            train_rows = random.Random(args.seed).sample(train_rows, min(args.max_train_samples, len(train_rows)))
        encoded = [e for e in (encode_example(tokenizer, r) for r in train_rows) if e]
        print(f"train examples: {len(encoded)} ({len(train_rows) - len(encoded)} skipped as longer than "
              f"{args.max_len} tokens)")

        if args.load_4bit:
            model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
        model = get_peft_model(model, LoraConfig(r=args.lora_r, lora_alpha=2 * args.lora_r, lora_dropout=0.05,
                                                 task_type="CAUSAL_LM", target_modules=LORA_TARGETS))
        for p in model.parameters():
            if p.requires_grad:
                p.data = p.data.float()  # fp32 adapters keep fp16 mixed precision from overflowing
        model.print_trainable_parameters()

        dev_rows = read_split("dev")
        if args.dev_samples and args.dev_samples < len(dev_rows):
            dev_rows = random.Random(args.seed).sample(dev_rows, args.dev_samples)
        callback = DevQuadF1(tokenizer, dev_rows, out_dir, args.patience) if args.dev_samples else None

        steps_per_epoch = math.ceil(len(encoded) / (args.batch_size * args.grad_accum))
        extra = {}
        if "group_by_length" in inspect.signature(transformers.TrainingArguments.__init__).parameters:
            extra["group_by_length"] = True  # less padding; removed in transformers 5
        training_args = transformers.TrainingArguments(
            output_dir=os.path.join(out_dir, "trainer"), num_train_epochs=args.epochs,
            per_device_train_batch_size=args.batch_size, gradient_accumulation_steps=args.grad_accum,
            learning_rate=args.lr, lr_scheduler_type="cosine",
            warmup_steps=max(1, int(0.03 * steps_per_epoch * args.epochs)), weight_decay=0.0,
            logging_steps=25, save_strategy="no", report_to="none", seed=args.seed,
            fp16=dtype == torch.float16, bf16=dtype == torch.bfloat16,
            gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False},
            optim="paged_adamw_8bit" if args.load_4bit else "adamw_torch",
            remove_unused_columns=False, dataloader_pin_memory=False, **extra)
        trainer = transformers.Trainer(model=model, args=training_args, train_dataset=encoded,
                                       data_collator=lambda batch: collate(batch, tokenizer.pad_token_id),
                                       callbacks=[callback] if callback else None)
        train_sec = trainer.train().metrics["train_runtime"]
        model.save_pretrained(os.path.join(out_dir, "last"))
        train_info = {"epochs": args.epochs, "epochs_run": round(trainer.state.epoch, 2), "patience": args.patience,
                      "train_examples": len(encoded), "lora_r": args.lora_r, "load_4bit": args.load_4bit,
                      "dtype": str(dtype).replace("torch.", ""), "train_sec": train_sec}
        if callback:
            train_info.update(dev_history=callback.history, stopped_early=callback.stopped_early,
                              best_dev_quad_f1=callback.best_f1)
            if callback.best_state is not None:
                set_peft_model_state_dict(model, callback.best_state)
                print(f"best dev quad-F1 {callback.best_f1:.4f}; adapter saved to {os.path.join(out_dir, 'best')}")

    test_rows = read_split("test")
    with open(os.path.join(data_path, "processed", f"{args.dataset}_ground_truth.json"), "w", encoding="utf-8") as f:
        json.dump(test_rows, f, indent=4, ensure_ascii=False)
    t0 = time.perf_counter()
    preds = generate(model, tokenizer, [r["sentence"] for r in test_rows])
    infer_sec = time.perf_counter() - t0

    prefix = f"{tag}_ft_{args.dataset}"
    run_name = f"{prefix}_{next_run_index(prefix)}"
    with open(os.path.join(data_path, "processed", run_name + ".json"), "w", encoding="utf-8") as f:
        json.dump({"agent_ft": preds}, f, indent=4, ensure_ascii=False)
    timing_dir = os.path.join(data_path, "processed", "timing")
    os.makedirs(timing_dir, exist_ok=True)
    with open(os.path.join(timing_dir, run_name + "_timing.json"), "w", encoding="utf-8") as f:
        json.dump({"model": args.model, "task": "ft", "dataset": args.dataset, **train_info,
                   "n_instances": len(preds), "total_sec": infer_sec, "sec_per_instance": infer_sec / len(preds),
                   "n_api_calls": len(preds)}, f, indent=4)

    p, r, f1 = quad_f1(test_rows, preds)
    print(f"\nTEST quad-F1 {f1:.4f} (P {p:.4f} R {r:.4f}) on {len(preds)} rows")
    if train_info:
        print(f"training {train_info['train_sec']:.0f}s over {train_info['epochs_run']} epochs"
              f"{' (stopped early)' if train_info.get('stopped_early') else ''}, inference {infer_sec:.0f}s")
    print(f"predictions: data/processed/{run_name}.json")
    print(f"full report: python src/eval.py --model {args.model} --dataset {args.dataset} --tasks ft")


if __name__ == "__main__":
    main()
