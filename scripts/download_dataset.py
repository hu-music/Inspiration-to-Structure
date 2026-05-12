import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo_id", default="cszhu09876/ios-disco-abc")
    parser.add_argument("--output_dir", default="./data")
    parser.add_argument("--train_name", default="train_dataset_abc")
    parser.add_argument("--eval_name", default="eval_dataset_abc")
    parser.add_argument("--train_split", default="train")
    parser.add_argument("--eval_split", default="validation")
    return parser.parse_args()


def main():
    from datasets import load_dataset

    args = parse_args()
    dataset = load_dataset(args.repo_id)

    train_dir = os.path.join(args.output_dir, args.train_name)
    eval_dir = os.path.join(args.output_dir, args.eval_name)
    os.makedirs(args.output_dir, exist_ok=True)

    dataset[args.train_split].save_to_disk(train_dir)
    dataset[args.eval_split].save_to_disk(eval_dir)

    print(f"Saved train split to: {train_dir}")
    print(f"Saved eval split to: {eval_dir}")


if __name__ == "__main__":
    main()
