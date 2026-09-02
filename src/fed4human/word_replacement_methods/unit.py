import re

from gensim.models.keyedvectors import KeyedVectors

from fed4human.word_replacement_methods.base import BaseWordReplacementMethod
from fed4human.word_replacement_methods.morphological_analyzer import MorphologicalAnalyzer
from fed4human.word_replacement_methods.utils import get_similar_words_using_word2vec


class UnitWordReplacementMethod(BaseWordReplacementMethod):
    def __init__(self, w2v_model_path: str) -> None:
        self.morphological_analyzer = MorphologicalAnalyzer()
        self.w2v_model = KeyedVectors.load_word2vec_format(w2v_model_path, binary=False)

        self.katakana_pattern = re.compile("[\u30a1-\u30ff]+")  # \u30A1-\u30FF: katakana

    def get_conversation_to_judge_unit(self, target_word: str, repalcement: str) -> bool:
        target_word_length = len(target_word)
        repalcement_length = len(repalcement)
        if target_word_length >= repalcement_length and target_word[:repalcement_length] == repalcement:
            return False
        elif target_word_length <= repalcement_length and target_word == repalcement[:target_word_length]:
            return False
        else:
            return True

    def get_target_word2replacement(self, text: str) -> dict[str, str]:
        morpheme2feature_dict = self.morphological_analyzer.get_morpheme2feature_dict(text)
        numeral_classifiers = list(
            set(
                [
                    morpheme
                    for morpheme, feature_dict in morpheme2feature_dict.items()
                    if "助数詞可能" in feature_dict.subpos and self.katakana_pattern.match(morpheme)
                ]
            )
        )

        target_word2replacement = {}
        for numeral_classifier in numeral_classifiers:
            replacement_candidates = get_similar_words_using_word2vec(numeral_classifier, self.w2v_model)
            replacements = [
                rc
                for rc in replacement_candidates
                if self.get_conversation_to_judge_unit(numeral_classifier, rc) and self.katakana_pattern.fullmatch(rc)
                # if numeral_classifier not in rc and self.katakana_pattern.fullmatch(rc)
            ]
            if len(replacements) > 0:
                target_word2replacement[numeral_classifier] = replacements[0]
        return target_word2replacement
