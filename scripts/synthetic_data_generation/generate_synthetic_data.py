import json
from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path
from unicodedata import normalize

from tqdm import tqdm

from fed4human.utils import set_seed
from fed4human.word_replacement_methods.constants import Category
from fed4human.word_replacement_methods.utils import category_type, get_word_replacement_method


def validate_categories(categories: list[Category]) -> None:
    if Category.ANTONYM in categories and (
        Category.NAMED_ENTITY in categories or Category.KANJI_MISCONVERSION in categories
    ):
        raise ValueError("cannot specify antonym and named_entity/kanji_misconversion simultaneously")


def main():
    parser = ArgumentParser(description="script to generate synthetic data from Wikinews")
    parser.add_argument("CONFIG", type=Path, help="path to config (./config/synthetic_data_generation.json)")
    parser.add_argument("OUT_ROOT", type=Path, help="path to output root")
    parser.add_argument(
        "--categories",
        nargs="*",
        type=category_type,
        required=True,
        help=f'categories of human-induced factual errors ({", ".join(c.value for c in Category)})',
    )
    args = parser.parse_args()

    set_seed(0)

    validate_categories(args.categories)

    with args.CONFIG.open(mode="r") as fin:
        config = json.load(fin)

    with open(config["raw_data_path"], mode="r") as fin:
        raw_examples = [json.loads(line) for line in fin]

    for category in args.categories:
        wrm = get_word_replacement_method(category, config)
        out_dir = args.OUT_ROOT / category.value
        out_dir.mkdir(parents=True, exist_ok=True)
        for raw_example in tqdm(raw_examples, desc=f"generating synthetic data of {category.value}"):
            # MeCab removes whitespace internally
            normalized_text = normalize("NFKC", raw_example["text"]).replace(" ", "").replace("\n", "")
            if category == Category.DATE:
                try:
                    publish_date = datetime.strptime(raw_example["datetime"], "%Y-%m-%dT%H:%M:%SZ")
                except ValueError:
                    publish_date = None
                target_word2replacement = wrm.get_target_word2replacement(normalized_text, publish_date)
            else:
                target_word2replacement = wrm.get_target_word2replacement(normalized_text)

            for target_word, value in target_word2replacement.items():
                if category == Category.NAMED_ENTITY:
                    replacement = value["replacement"]
                    subcategory = value["subcategory"]
                else:
                    replacement = value
                    subcategory = ""
                synthetic_example = {
                    "source_id": raw_example["id"],
                    "datetime": raw_example["datetime"],
                    "category": category.value,
                    "subcategory": subcategory,
                    "text": normalized_text,
                    "target_word": target_word,
                    "replacement": replacement,
                }
                with (out_dir / "synthetic_examples.jsonl").open(mode="a", encoding="utf-8") as fout:
                    fout.write(json.dumps(synthetic_example, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
