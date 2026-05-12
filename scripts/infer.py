import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--prompt", default=None, help="Free-form user prompt. Overrides --abc/--abc_file + --section.")
    parser.add_argument("--system_prompt", default="You are a music composer assistant. Given ABC notation input, your task is to generate music in ABC format.")
    parser.add_argument("--abc", default=None)
    parser.add_argument("--abc_file", default=None)
    parser.add_argument("--section", default=None)
    parser.add_argument("--max_seq_length", type=int, default=8192)
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--output_file", default=None)
    parser.add_argument("--load_in_4bit", action="store_true", default=True)
    parser.add_argument("--no_4bit", dest="load_in_4bit", action="store_false")
    return parser.parse_args()


def read_abc(args):
    if args.abc_file:
        with open(args.abc_file, "r", encoding="utf-8") as f:
            return f.read().strip()
    if args.abc:
        return args.abc.strip()
    raise ValueError("Provide either --prompt, --abc, or --abc_file")


def main():
    args = parse_args()

    import torch
    from unsloth import FastLanguageModel
    from unsloth.chat_templates import get_chat_template

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_path,
        max_seq_length=args.max_seq_length,
        dtype=None,
        load_in_4bit=args.load_in_4bit,
    )
    tokenizer = get_chat_template(tokenizer, chat_template="llama-3.1")
    FastLanguageModel.for_inference(model)

    if args.prompt:
        user_content = args.prompt.strip()
    else:
        if not args.section:
            raise ValueError("--section is required when --prompt is not provided")
        abc = read_abc(args)
        user_content = f"Given:{abc}, generate {args.section}."

    messages = [
        {
            "role": "system",
            "content": args.system_prompt,
        },
        {
            "role": "user",
            "content": user_content,
        },
    ]

    input_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            input_ids=input_ids,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            do_sample=args.temperature > 0,
            use_cache=True,
        )

    generated = tokenizer.decode(output_ids[0][input_ids.shape[-1] :], skip_special_tokens=False)
    print(generated)

    if args.output_file:
        os.makedirs(os.path.dirname(args.output_file) or ".", exist_ok=True)
        with open(args.output_file, "w", encoding="utf-8") as f:
            f.write(generated)


if __name__ == "__main__":
    main()
