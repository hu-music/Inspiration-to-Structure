import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ios_disco.data import inspect_dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--split", default=None)
    parser.add_argument("--max_scan", type=int, default=None)
    args = parser.parse_args()

    stats = inspect_dataset(args.data_path, split=args.split, max_scan=args.max_scan)

    print("Num rows scanned:", stats["num_rows_scanned"])
    print("Columns:", stats["columns"])
    print("\nRole counts:")
    print(stats["role_counts"])
    print("\nABC block count per sample:")
    print(stats["block_count_per_sample"])
    print("\nSection label counts:")
    print(stats["section_counts"])
    print("\nLikely usable for DiSCO:")
    print("3-block intra-like samples:", stats["intra_like_samples"])
    print("4-block inter-like samples:", stats["inter_like_samples"])
    print("samples without ABC blocks:", stats["samples_without_abc_blocks"])

    print("\nDecision:")
    if stats["intra_like_samples"] and stats["inter_like_samples"]:
        print("YES: usable for both intra and inter DiSCO training.")
    elif stats["intra_like_samples"]:
        print("PARTIAL: usable for intra DiSCO training.")
    elif stats["inter_like_samples"]:
        print("PARTIAL: usable for inter DiSCO training.")
    else:
        print("NO: likely only usable for SFT unless triplets are reconstructed.")


if __name__ == "__main__":
    main()
