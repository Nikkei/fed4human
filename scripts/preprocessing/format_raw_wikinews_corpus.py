import json
from argparse import ArgumentParser
from pathlib import Path

from tqdm import tqdm


def main():
    parser = ArgumentParser(description="script to format raw WikiNews corpus")
    parser.add_argument("IN_DIR", type=Path, help="path to input directory")
    parser.add_argument("TITLE2TIMESTAMP", type=Path, help="path to title2timestamp.json")
    parser.add_argument("OUT_FILE", type=Path, help="path to output file")
    args = parser.parse_args()

    min_len = 100
    with args.TITLE2TIMESTAMP.open(mode="r") as fin:
        title2timestamp = json.load(fin)

    out_objs = []
    for in_file in tqdm(args.IN_DIR.rglob("wiki_*"), desc="for loop of in_dir.glob"):
        with in_file.open(mode="r") as fin:
            for line in tqdm(fin, desc="for loop of fin"):
                in_obj = json.loads(line)
                text = in_obj["text"]
                if (timestamp := title2timestamp.get(in_obj["title"])) and len(text) >= min_len:
                    in_obj["datetime"] = timestamp
                    cleaned = text.replace('"英語版ウィキニュースからの翻訳です。"', "")
                    cleaned = cleaned.replace('"英語版ウィキニュースの翻訳を含みます。"', "")
                    out_objs.append(
                        {
                            "id": in_obj["id"],
                            "revid": in_obj["revid"],
                            "url": in_obj["url"],
                            "title": in_obj["title"],
                            "datetime": in_obj["datetime"],
                            "text": cleaned,
                        }
                    )
    print(f"num articles: {len(out_objs)}")

    with args.OUT_FILE.open(mode="w") as fout:
        for out_obj in out_objs:
            fout.write(json.dumps(out_obj, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
