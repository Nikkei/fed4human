import random

import mozcpy
import regex
import spacy
from namedivider import GBDTNameDivider
from spacy.lang.ja import Japanese

from fed4human.word_replacement_methods.base import BaseWordReplacementMethod
from fed4human.word_replacement_methods.morphological_analyzer import Transliterator


class KanjiMisconversionWordReplacementMethod(BaseWordReplacementMethod):
    def __init__(self, ner_model_name: str) -> None:
        spacy.prefer_gpu()
        self.nlp: Japanese = spacy.load(ner_model_name)
        self.gbdt_name_divider = GBDTNameDivider()
        self.transliterator = Transliterator()
        self.converter = mozcpy.Converter()

        self.kanji_only_pattern = regex.compile(r"\p{Script=Han}+")

    def get_target_word2replacement(self, text: str) -> dict[str, str]:
        person_names = self.extract_person_names(text)

        target_word2replacement = {}
        for person_name in person_names:
            if self.kanji_only_pattern.fullmatch(person_name):
                if len(person_name) > 3:
                    index = random.randint(0, 1)
                    family_and_given_names = self.divide_name_into_family_and_given_names(person_name)
                    hiragana = self.transliterator.transliterate_text(family_and_given_names[index])
                    replacement_candidates = self.convert_hiragana_into_kanji(hiragana)
                    replacements = [
                        rc
                        for rc in replacement_candidates
                        if rc != family_and_given_names[index] and self.kanji_only_pattern.fullmatch(rc)
                    ]
                    if len(replacements) > 0:
                        family_and_given_names[index] = replacements[0]
                        target_word2replacement[person_name] = "".join(family_and_given_names)
                elif len(person_name) > 1:
                    hiragana = self.transliterator.transliterate_text(person_name)
                    replacement_candidates = self.convert_hiragana_into_kanji(hiragana)
                    replacements = [
                        rc
                        for rc in replacement_candidates
                        if rc != person_name and self.kanji_only_pattern.fullmatch(rc)
                    ]
                    if len(replacements) > 0:
                        target_word2replacement[person_name] = replacements[0]
        return target_word2replacement

    def extract_person_names(self, text: str) -> list[str]:
        doc = self.nlp(text)
        person_names = []
        for ent in doc.ents:
            if ent.label_ == "Person" and ent.text not in person_names:
                person_names.append(ent.text)
        return person_names

    def divide_name_into_family_and_given_names(self, name: str) -> list[str]:
        divided_name = self.gbdt_name_divider.divide_name(name)
        return [divided_name.family, divided_name.given]

    def convert_hiragana_into_kanji(self, text: str) -> list[str]:
        return self.converter.convert(text, n_best=10)
