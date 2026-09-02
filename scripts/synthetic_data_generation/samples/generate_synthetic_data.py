import json
from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path
from unicodedata import normalize

from fed4human.utils import set_seed
from fed4human.word_replacement_methods.constants import Category
from fed4human.word_replacement_methods.utils import category_type, get_word_replacement_method


def main():
    parser = ArgumentParser(description="sample script to generate synthetic data")
    parser.add_argument("CONFIG", type=Path, help="path to config (./config/synthetic_data_generation.json)")
    parser.add_argument(
        "--category",
        type=category_type,
        required=True,
        help=f'category (any of {", ".join(c.value for c in Category)})',
    )
    args = parser.parse_args()

    with args.CONFIG.open(mode="r") as fin:
        config = json.load(fin)

    set_seed(0)

    wrm = get_word_replacement_method(args.category, config)

    samples = [
        {
            "text": "2021年7月23日に日本でオリンピックが開催されました。",
            "publish_date": datetime.strptime("2021-07-25 13:03:41+00:00", "%Y-%m-%d %H:%M:%S%z"),
        },
        {
            "text": "2025年の中国の国家予算は約115兆2000億円である。",
            "publish_date": datetime.strptime("2025-07-13 13:03:41+00:00", "%Y-%m-%d %H:%M:%S%z"),
        },
        {
            "text": "ナイル川の全長は約6,695kmとされています。",
            "publish_date": datetime.strptime("2027-01-01 13:03:41+00:00", "%Y-%m-%d %H:%M:%S%z"),
        },
        {
            "text": "5日の日経株価は上昇傾向である",
            "publish_date": datetime.strptime("2019-07-05 13:03:41+00:00", "%Y-%m-%d %H:%M:%S%z"),
        },
        {
            "text": "1メートル間隔で電柱が設置されている",
            "publish_date": datetime.strptime("2025-08-06 13:03:41+00:00", "%Y-%m-%d %H:%M:%S%z"),
        },
        {
            "text": "物理学者の湯川秀樹が日本人として初めてノーベル賞を受賞した",
            "publish_date": datetime.strptime("2027-01-01 13:03:41+00:00", "%Y-%m-%d %H:%M:%S%z"),
        },
    ]

    for sample in samples:
        normalized_text = normalize("NFKC", sample["text"]).replace(" ", "").replace("\n", "")
        if args.category == Category.DATE:
            target_word2replacement = wrm.get_target_word2replacement(normalized_text, sample["publish_date"])
        else:
            target_word2replacement = wrm.get_target_word2replacement(normalized_text)
        if target_word2replacement:
            print(f"original: {normalized_text}")
            for target_word, value in target_word2replacement.items():
                if isinstance(value, dict):
                    replaced = normalized_text.replace(target_word, value["replacement"])
                else:
                    replaced = normalized_text.replace(target_word, value)
                print(f"replaced: {replaced}")
            print()


if __name__ == "__main__":
    main()
