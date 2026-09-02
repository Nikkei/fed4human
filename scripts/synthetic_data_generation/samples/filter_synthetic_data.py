import json
from argparse import ArgumentParser
from pathlib import Path

from fed4human.utils import set_seed
from fed4human.word_replacement_methods.evaluator import Evaluator


def main():
    parser = ArgumentParser(description="sample script to filter target word and replacement pairs using LLM")
    parser.add_argument("CONFIG", type=Path, help="path to config (./config/synthetic_data_generation.json)")
    args = parser.parse_args()

    with args.CONFIG.open(mode="r") as fin:
        config = json.load(fin)

    set_seed(0)

    evaluator = Evaluator(config["pretrained_llm_name_or_path"])

    samples = [
        {"text": "欧州の景気が悪い", "target_word": "欧州", "replacement": "ヨーロッパ"},
        {"text": "日本の景気が良い", "target_word": "日本", "replacement": "中国"},
    ]

    outputs = evaluator.judge_synonym(samples)

    for sample, output in zip(samples, outputs):
        try:
            obj = json.loads(output)
            answer = obj["answer"]
        except json.decoder.JSONDecodeError:
            print("JSONDecodeError")
            continue
        if answer is False:
            print(f'{sample["target_word"]} and {sample["replacement"]} are not judged synonymous')
        else:
            print(f'{sample["target_word"]} and {sample["replacement"]} are judged synonymous')


if __name__ == "__main__":
    main()
