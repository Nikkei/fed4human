import json
import random
from argparse import ArgumentParser
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from fed4human.utils import set_seed
from fed4human.word_replacement_methods.constants import TARGET_NE_CATEGORIES, Category
from fed4human.word_replacement_methods.utils import until_type


def load_synthetic_examples(in_root: Path, category: Category) -> list[dict[str, str]]:
    if category in {Category.NAMED_ENTITY, Category.ANTONYM}:
        basename = "filtered.jsonl"
    else:
        basename = "synthetic_examples.jsonl"
    in_file = in_root / category.value / basename
    with in_file.open(mode="r") as fin:
        synthetic_examples = [json.loads(line) for line in fin]
    return synthetic_examples


def sample_examples(examples: list[dict[str, str]], num_samples: int) -> list[dict[str, str]]:
    if len(examples) < num_samples:
        return examples
    else:
        return random.sample(examples, num_samples)


def get_test_samples(in_root: Path, num_samples: int, until: datetime) -> list[dict[str, str]]:
    positive_samples = []
    negative_examples = []
    appeared = set()
    for category in Category:
        synthetic_examples = load_synthetic_examples(in_root, category)
        for synthetic_example in synthetic_examples:
            publish_date = datetime.strptime(synthetic_example["datetime"], "%Y-%m-%dT%H:%M:%S%z")
            if publish_date > until:
                continue
            if synthetic_example["source_id"] not in appeared:
                negative_examples.append(
                    {
                        "source_id": synthetic_example["source_id"],
                        "datetime": synthetic_example["datetime"],
                        "category": "negative",
                        "subcategory": "",
                        "text": synthetic_example["text"],
                        "target_word": "",
                        "replacement": "",
                    }
                )
                appeared.add(synthetic_example["source_id"])

        if category == Category.NAMED_ENTITY:
            for target_ne_category in TARGET_NE_CATEGORIES:
                synthetic_ne_examples = [e for e in synthetic_examples if e["subcategory"] == target_ne_category]
                positive_samples += sample_examples(synthetic_ne_examples, num_samples)
        else:
            positive_samples += sample_examples(synthetic_examples, num_samples)
    negative_samples = sample_examples(negative_examples, num_samples)
    test_samples = positive_samples + negative_samples
    return test_samples


def truncate_text(example: dict[str, str], max_length: int) -> dict[str, str] | None:
    if len(example["text"]) > max_length:
        sentences = example["text"].split("。")
        sent_cum_lens = np.array([len(s) for s in sentences]).cumsum()
        index = (sent_cum_lens <= max_length).sum().item()
        truncated_text = "。".join(sentences[:index])
        if example["target_word"] not in truncated_text:
            return None
        example["text"] = truncated_text
    return example


def get_train_samples(
    in_root: Path,
    num_train_examples: int,
    test_sample_source_ids: set[int],
    max_length: int,
) -> list[dict[str, str]]:
    positive_samples = []
    negative_examples = []
    appeared = set()
    for category in Category:
        synthetic_examples = load_synthetic_examples(in_root, category)
        synthetic_examples = list(filter(lambda x: truncate_text(x, max_length), synthetic_examples))
        for synthetic_example in synthetic_examples:
            if synthetic_example["source_id"] in test_sample_source_ids:
                continue
            if synthetic_example["source_id"] not in appeared:
                negative_examples.append(
                    {
                        "source_id": synthetic_example["source_id"],
                        "datetime": synthetic_example["datetime"],
                        "category": "negative",
                        "subcategory": "",
                        "text": synthetic_example["text"],
                        "target_word": "",
                        "replacement": "",
                    }
                )
                appeared.add(synthetic_example["source_id"])

        if category == Category.NAMED_ENTITY:
            for target_ne_category in TARGET_NE_CATEGORIES:
                synthetic_ne_examples = [e for e in synthetic_examples if e["subcategory"] == target_ne_category]
                positive_samples += sample_examples(synthetic_ne_examples, num_train_examples)
        else:
            positive_samples += sample_examples(synthetic_examples, num_train_examples)
    train_samples = positive_samples + negative_examples
    return train_samples


def get_few_shot_examples(
    in_root: Path,
    num_few_shot_examples: int,
    test_sample_source_ids: set[int],
    until: datetime,
    max_length: int,
) -> list[dict[str, str]]:
    few_shot_examples = []
    for category in Category:
        synthetic_examples = load_synthetic_examples(in_root, category)

        filtered = []
        for synthetic_example in synthetic_examples:
            publish_date = datetime.strptime(synthetic_example["datetime"], "%Y-%m-%dT%H:%M:%S%z")
            if (
                synthetic_example["source_id"] in test_sample_source_ids
                or publish_date > until
                or len(synthetic_example["text"]) > max_length
            ):
                continue
            filtered.append(synthetic_example)

        few_shot_examples.append(random.choice(filtered))
    return random.sample(few_shot_examples, num_few_shot_examples)


def main():
    parser = ArgumentParser(description="script to get test samples and few-shot examples")
    parser.add_argument("IN_ROOT", type=Path, help="path to input root")
    parser.add_argument("OUT_DIR", type=Path, help="path to output directory")
    parser.add_argument(
        "--until",
        type=until_type,
        default=datetime.strptime("2024-01-01", "%Y-%m-%d").astimezone(timezone.utc),
        help="cutoff date (%Y-%m-%d)",
    )
    args = parser.parse_args()

    set_seed(0)

    num_samples = 100
    num_few_shot_examples = 6
    num_train_examples = 2000
    train_max_length = 500
    few_shot_max_length = 300

    test_samples = get_test_samples(args.IN_ROOT, num_samples, args.until)
    with (args.OUT_DIR / "test_samples.jsonl").open(mode="w", encoding="utf-8") as fout:
        for sample in test_samples:
            fout.write(json.dumps(sample, ensure_ascii=False) + "\n")

    test_sample_source_ids = {sample["source_id"] for sample in test_samples}
    few_shot_examples = get_few_shot_examples(
        args.IN_ROOT, num_few_shot_examples, test_sample_source_ids, args.until, few_shot_max_length
    )
    with (args.OUT_DIR / "few-shot_examples.jsonl").open(mode="w", encoding="utf-8") as fout:
        for example in few_shot_examples:
            fout.write(json.dumps(example, ensure_ascii=False) + "\n")

    trian_samples = get_train_samples(args.IN_ROOT, num_train_examples, test_sample_source_ids, train_max_length)
    with (args.OUT_DIR / "train_samples.jsonl").open(mode="w", encoding="utf-8") as fout:
        for sample in trian_samples:
            fout.write(json.dumps(sample, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
