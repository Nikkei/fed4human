import json
from dataclasses import dataclass
from enum import IntEnum

from MeCab import Tagger
from pykakasi import kakasi
from unidic_lite import DICDIR


class Transliterator:
    def __init__(self) -> None:
        self.kks = kakasi()

    def transliterate_text(self, text: str) -> str:
        result = self.kks.convert(text)
        return "".join(word["hira"] for word in result)


@dataclass
class FeatureDict:
    pos: str
    subpos: tuple[str, str, str]
    conjform: str
    lemma: str
    reading: str


class Index(IntEnum):
    POS = 0
    SUBPOS1 = 1
    SUBPOS2 = 2
    SUBPOS3 = 3
    CONJTYPE = 4  # 活用型
    CONJFORM = 5  # 活用形
    LEMMA = 6  # 原形
    READING = 7  # 読み
    PRONUNCIATION = 8  # 発音


class MorphologicalAnalyzer:
    def __init__(self) -> None:
        self.transliterator = Transliterator()
        self.tagger = Tagger(f"-r /dev/null -d {DICDIR}")

        di = self.tagger.dictionary_info()
        print("mecab-py dictionary info:")
        while di:
            print(json.dumps({"version": di.version, "filename": di.filename}))
            di = di.next

    def segment_text_into_morphemes(self, text: str) -> list[str]:
        node = self.tagger.parseToNode(text)

        morphemes = []
        while node:
            morpheme = node.surface
            if morpheme:
                morphemes.append(morpheme)
            node = node.next
        return morphemes

    @staticmethod
    def get_morpheme_index2char_index(text: str, morphemes: list[str]) -> list[int]:
        morpheme_index2char_index = []
        offset = 0
        for morpheme in morphemes:
            morpheme_index2char_index.append(offset + text[offset:].index(morpheme))
            offset += len(morpheme)
        return morpheme_index2char_index

    def get_morpheme2feature_dict(self, text: str) -> dict[str, FeatureDict]:
        node = self.tagger.parseToNode(text)

        morpheme2feature_dict = {}
        while node:
            morpheme = node.surface
            if morpheme not in morpheme2feature_dict.keys():
                features = node.feature.split(",")
                morpheme2feature_dict[morpheme] = FeatureDict(
                    pos=features[Index.POS],
                    subpos=(
                        features[Index.SUBPOS1],
                        features[Index.SUBPOS2],
                        features[Index.SUBPOS3],
                    ),
                    conjform=features[Index.CONJFORM],
                    lemma=features[Index.LEMMA] if len(features) >= 7 else "*",
                    reading=self.transliterator.transliterate_text(morpheme),
                )
            node = node.next
        return morpheme2feature_dict
