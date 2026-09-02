import json
import xml.etree.ElementTree as ET
from argparse import ArgumentParser
from pathlib import Path

from tqdm import tqdm


def main():
    parser = ArgumentParser(description="script to get title2timestamp")
    parser.add_argument("IN_FILE", type=Path, help="path to input file")
    parser.add_argument("OUT_FILE", type=Path, help="path to output file")
    args = parser.parse_args()

    ns = "{http://www.mediawiki.org/xml/export-0.11/}"
    title2timestamp = {}
    for _, elem in tqdm(ET.iterparse(args.IN_FILE, events=("end",)), desc="for loop of ET.iterparse"):
        if elem.tag == ns + "page":
            title = elem.find(ns + "title").text
            revision = elem.find(ns + "revision")
            if revision is not None:
                timestamp = revision.find(ns + "timestamp").text
                title2timestamp[title] = timestamp
                print(f"{title}: {timestamp}")
            elem.clear()
    with args.OUT_FILE.open(mode="w") as fout:
        json.dump(title2timestamp, fout, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
