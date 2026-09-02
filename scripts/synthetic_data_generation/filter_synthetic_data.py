import json
from argparse import ArgumentParser
from pathlib import Path

from fed4human.word_replacement_methods.evaluator import Evaluator
from fed4human.word_replacement_methods.utils import category_type


def main():
    parser = ArgumentParser(description="script to filter target word and replacement pairs using LLM")
    parser.add_argument("IN_ROOT", type=Path, help="path to input root")
    parser.add_argument("CONFIG", type=Path, help="path to config (./config/synthetic_data_generation.json)")
    parser.add_argument("--category", type=category_type, help="category of human-induced factual errors")
    args = parser.parse_args()

    in_file = args.IN_ROOT / args.category.value / "synthetic_examples.jsonl"
    with in_file.open(mode="r", encoding="utf-8") as fin:
        synthetic_examples = [json.loads(line) for line in fin]

    with args.CONFIG.open(mode="r") as fin:
        config = json.load(fin)

    evaluator = Evaluator(config["pretrained_llm_name_or_path"])

    outputs = evaluator.judge_synonym(synthetic_examples)

    filtered = []
    for synthetic_example, output in zip(synthetic_examples, outputs):
        completion = output.outputs[0].text  # 0: top-1 result
        try:
            obj = json.loads(completion)
            answer = obj["answer"]
        except json.decoder.JSONDecodeError:
            continue
        if answer is False:  # retain if a target word and its replacement are not judged synonymous
            filtered.append(synthetic_example)
        else:
            print(f'{synthetic_example["target_word"]} and {synthetic_example["replacement"]} are judged synonymous')

    out_file = args.IN_ROOT / args.category.value / "filtered.jsonl"
    with out_file.open(mode="w", encoding="utf-8") as fout:
        for synthetic_example in filtered:
            fout.write(json.dumps(synthetic_example, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
