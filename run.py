"""
Inference entry point.

Usage:
    python run.py --weights ./weights --prompt "Tell me about RoPE"
    python run.py --weights ./weights --chat  # interactive chat loop
"""
import argparse
import sys
from pathlib import Path

import torch

from baseline import load_from_hf_dir
from tokenizer import Qwen3Tokenizer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--weights", default="weights", help="Path to downloaded weight directory")
    p.add_argument("--prompt", default=None, help="Single prompt (non-chat)")
    p.add_argument("--chat", action="store_true", help="Interactive chat loop")
    p.add_argument("--max_new_tokens", type=int, default=512)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top_p", type=float, default=0.9)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--dtype", default="bfloat16", choices=["float32", "float16", "bfloat16"])
    return p.parse_args()


def load(args: argparse.Namespace):
    weights_dir = Path(args.weights)
    if not weights_dir.exists():
        sys.exit(f"Weights not found at {weights_dir}. Run download_weights.py first.")

    print(f"Loading model from {weights_dir} on {args.device} ({args.dtype}) ...")
    dtype = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[args.dtype]

    model = load_from_hf_dir(weights_dir, device=args.device)
    model = model.to(dtype).eval()

    tokenizer = Qwen3Tokenizer.from_dir(weights_dir)
    print("Ready.")
    return model, tokenizer


def generate(model, tokenizer: Qwen3Tokenizer, args, messages: list[dict]) -> str:
    input_ids = tokenizer.apply_chat_template(messages)
    ids_tensor = torch.tensor([input_ids], device=args.device)

    out_ids = model.generate(
        ids_tensor,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        eos_token_id=Qwen3Tokenizer.EOS_ID,
    )

    new_ids = out_ids[0, len(input_ids):].tolist()
    return tokenizer.decode(new_ids)


def main():
    args = parse_args()
    model, tokenizer = load(args)

    if args.chat:
        history: list[dict] = []
        print("Chat mode — type 'quit' to exit.\n")
        while True:
            user_input = input("You: ").strip()
            if user_input.lower() in ("quit", "exit", "q"):
                break
            history.append({"role": "user", "content": user_input})
            response = generate(model, tokenizer, args, history)
            print(f"Assistant: {response}\n")
            history.append({"role": "assistant", "content": response})
    elif args.prompt:
        messages = [{"role": "user", "content": args.prompt}]
        print(generate(model, tokenizer, args, messages))
    else:
        print("Provide --prompt or --chat. See --help.")


if __name__ == "__main__":
    main()
