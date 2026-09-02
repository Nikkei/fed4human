from argparse import ArgumentParser
from pathlib import Path

from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    parser = ArgumentParser(description="script to merge an adapter into a base model")
    parser.add_argument("IN_DIR", type=Path, help="path to input directory that contains an adapter")
    parser.add_argument("BASE_MODEL", type=Path, help="path to a base model (e.g., google/gemma-4-E4B-it_0)")
    parser.add_argument("OUT_DIR", type=Path, help="path to output directory")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.BASE_MODEL)
    tokenizer.save_pretrained(args.OUT_DIR)

    base_model = AutoModelForCausalLM.from_pretrained(args.BASE_MODEL, dtype="bfloat16", device_map=None)
    adapter = PeftModel.from_pretrained(base_model, args.IN_DIR)
    adapter = adapter.merge_and_unload()
    adapter.save_pretrained(args.OUT_DIR)


if __name__ == "__main__":
    main()
