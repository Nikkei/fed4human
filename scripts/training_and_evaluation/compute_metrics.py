import json
from argparse import ArgumentParser
from pathlib import Path

from fed4human.deep_learning.metrics import Metric


def main():
    parser = ArgumentParser(description="script to compute metrics")
    parser.add_argument("IN_DIR", type=Path, help="path to input directory")
    parser.add_argument("--out-dir", type=Path, default=None, help="path to output directory")
    args = parser.parse_args()

    in_files = [
        args.IN_DIR / f"{data_source}_{model_id}_{setting}.jsonl"
        for data_source in ["wikinews", "nikkei"]
        for model_id in ["gpt-5.4", "Qwen3-Swallow-8B-SFT-v0p2", "gemma-4-E4B-it"]
        for setting in ["fewshot", "qlora", "fullpara"]
    ]

    if args.out_dir:
        args.out_dir.mkdir(parents=True, exist_ok=True)

    for in_file in in_files:
        if in_file.exists() is False:
            continue
        metric = Metric()
        with in_file.open(mode="r") as fin:
            for line in fin:
                in_obj = json.loads(line)
                metric.update_sentence_level_outcomes(in_obj)
                metric.update_word_level_outcomes(in_obj)
        print(f"** {in_file.stem} **")
        outcomes = metric.category2sentence_level_outcomes["micro"]
        print(f"micro/sent/prec: {outcomes.compute_precision() * 100:.02f} ({outcomes.TP + outcomes.FP})")
        print(f"micro/sent/rec: {outcomes.compute_recall() * 100:.02f} ({outcomes.TP + outcomes.FN})")
        print(f"micro/sent/f1: {outcomes.compute_f1_score() * 100:.02f} ({outcomes.TP + outcomes.FN})")
        outcomes = metric.category2word_level_outcomes["micro"]
        print(f"micro/word/prec: {outcomes.compute_precision() * 100:.02f} ({outcomes.TP + outcomes.FP})")
        print(f"micro/word/rec: {outcomes.compute_recall() * 100:.02f} ({outcomes.TP + outcomes.FN})")
        print(f"micro/word/f1: {outcomes.compute_f1_score() * 100:.02f} ({outcomes.TP + outcomes.FN})")
        if args.out_dir:
            with (args.out_dir / in_file.name).open(mode="w") as fout:
                for (
                    category,
                    difficulty2sentence_level_outcomes,
                ) in metric.category2difficulty2sentence_level_outcomes.items():
                    for difficulty, outcomes in difficulty2sentence_level_outcomes.items():
                        out_obj = {
                            "category": category,
                            "difficulty": difficulty,
                            "f1": round(outcomes.compute_f1_score(), 2),
                        }
                        fout.write(json.dumps(out_obj, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
