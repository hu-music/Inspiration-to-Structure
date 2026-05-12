import argparse
import os
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--train_dataset", default="cszhu09876/ios-disco-abc")
    parser.add_argument("--eval_dataset", default=None)
    parser.add_argument("--train_split", default="train")
    parser.add_argument("--eval_split", default="validation")
    parser.add_argument("--model_name", default="unsloth/Llama-3.2-1B-Instruct")
    parser.add_argument("--output_dir", default="outputs/ios_disco")
    parser.add_argument("--max_seq_length", type=int, default=8192)
    parser.add_argument("--per_device_train_batch_size", type=int, default=4)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--num_train_epochs", type=float, default=1)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--warmup_steps", type=int, default=5)
    parser.add_argument("--logging_steps", type=int, default=100)
    parser.add_argument("--save_steps", type=int, default=1000)
    parser.add_argument("--save_strategy", default="steps", choices=["no", "steps", "epoch"])
    parser.add_argument("--optim", default="adamw_8bit")
    parser.add_argument("--lambda_contrastive", type=float, default=0.5)
    parser.add_argument("--gamma", type=float, default=0.1)
    parser.add_argument("--load_in_4bit", action="store_true", default=True)
    parser.add_argument("--no_4bit", dest="load_in_4bit", action="store_false")
    parser.add_argument("--dataset_num_proc", type=int, default=2)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--report_to", default="none")
    parser.add_argument("--log_contrastive_every", type=int, default=10)
    args = parser.parse_args()

    if args.config:
        with open(args.config, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        for key, value in cfg.items():
            if getattr(args, key, None) in (None, parser.get_default(key)):
                setattr(args, key, value)

    return args


def main():
    args = parse_args()

    import unsloth  # noqa: F401
    from unsloth import is_bfloat16_supported

    import torch
    from transformers import DataCollatorForSeq2Seq, TrainingArguments

    from ios_disco.data import load_dataset_any, select_split
    from ios_disco.modeling import load_unsloth_model
    from ios_disco.span_utils import default_end_patterns, default_start_patterns, find_abc_spans
    from ios_disco.trainer import DiSCOTrainer

    model, tokenizer = load_unsloth_model(
        model_name=args.model_name,
        max_seq_length=args.max_seq_length,
        load_in_4bit=args.load_in_4bit,
        for_training=True,
    )

    raw_train_dataset = load_dataset_any(args.train_dataset)
    train_dataset = select_split(raw_train_dataset, args.train_split)

    if args.eval_dataset:
        eval_dataset = select_split(load_dataset_any(args.eval_dataset), args.eval_split)
    elif hasattr(raw_train_dataset, "keys") and args.eval_split in raw_train_dataset:
        eval_dataset = select_split(raw_train_dataset, args.eval_split)
    else:
        eval_dataset = None

    start_patterns = default_start_patterns(tokenizer)
    end_patterns = default_end_patterns(tokenizer)

    response_start_ids = tokenizer.encode(
        "<|start_header_id|>assistant<|end_header_id|>\n\n",
        add_special_tokens=False,
    )
    eot_ids = tokenizer.encode("<|eot_id|>", add_special_tokens=False)

    def find_subsequence(sequence, pattern, start=0):
        if not pattern:
            return -1
        max_i = len(sequence) - len(pattern)
        for i in range(start, max_i + 1):
            if sequence[i : i + len(pattern)] == pattern:
                return i
        return -1

    def make_response_only_labels(input_ids):
        labels = [-100] * len(input_ids)
        cursor = 0
        while cursor < len(input_ids):
            header = find_subsequence(input_ids, response_start_ids, cursor)
            if header == -1:
                break
            answer_start = header + len(response_start_ids)
            answer_end = find_subsequence(input_ids, eot_ids, answer_start)
            if answer_end == -1:
                answer_end = len(input_ids)
            for j in range(answer_start, answer_end):
                labels[j] = input_ids[j]
            cursor = answer_end + max(1, len(eot_ids))
        return labels

    def tokenize_with_disco(example):
        ids = tokenizer(
            example["text"],
            add_special_tokens=False,
            truncation=True,
            max_length=args.max_seq_length,
        )["input_ids"]
        attention_mask = [1] * len(ids)
        labels = make_response_only_labels(ids)
        spans = find_abc_spans(ids, start_patterns, end_patterns, max_spans=4)
        starts = [-1, -1, -1, -1]
        ends = [-1, -1, -1, -1]
        for i, (start, end) in enumerate(spans[:4]):
            starts[i] = start
            ends[i] = end
        return {
            "input_ids": ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "disco_span_count": len(spans),
            "disco_span_starts": starts,
            "disco_span_ends": ends,
        }

    keep_columns = [
        "input_ids",
        "attention_mask",
        "labels",
        "disco_span_count",
        "disco_span_starts",
        "disco_span_ends",
    ]
    train_dataset = train_dataset.map(
        tokenize_with_disco,
        desc="Tokenizing with DiSCO spans",
        num_proc=args.dataset_num_proc,
        remove_columns=train_dataset.column_names,
    ).select_columns(keep_columns)
    if eval_dataset is not None:
        eval_dataset = eval_dataset.map(
            tokenize_with_disco,
            desc="Tokenizing eval with DiSCO spans",
            num_proc=args.dataset_num_proc,
            remove_columns=eval_dataset.column_names,
        ).select_columns(keep_columns)

    training_args = TrainingArguments(
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        warmup_steps=args.warmup_steps,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_strategy=args.save_strategy,
        optim=args.optim,
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=args.seed,
        output_dir=os.path.join(args.output_dir, "trainer_checkpoints"),
        report_to=args.report_to,
        remove_unused_columns=False,
    )

    trainer = DiSCOTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        max_seq_length=args.max_seq_length,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer),
        dataset_num_proc=args.dataset_num_proc,
        packing=False,
        args=training_args,
        lambda_contrastive=args.lambda_contrastive,
        gamma=args.gamma,
        log_contrastive_every=args.log_contrastive_every,
    )

    print("Starting training")
    trainer_stats = trainer.train()
    print(trainer_stats)

    os.makedirs(args.output_dir, exist_ok=True)
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Saved trained model/tokenizer to: {args.output_dir}")

    if torch.cuda.is_available():
        print("CUDA max memory allocated GB:", torch.cuda.max_memory_allocated() / 1024**3)


if __name__ == "__main__":
    main()
