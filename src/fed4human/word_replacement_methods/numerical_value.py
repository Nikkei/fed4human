import random

from fed4human.word_replacement_methods.base import BaseWordReplacementMethod
from fed4human.word_replacement_methods.constants import DATE_UNITS, NUMBER_PATTERN, NUMBER_UNITS
from fed4human.word_replacement_methods.morphological_analyzer import MorphologicalAnalyzer


class NumericalValueWordReplacementMethod(BaseWordReplacementMethod):
    def __init__(self) -> None:
        self.morphological_analyzer = MorphologicalAnalyzer()

    def get_target_word2replacement(self, text: str) -> dict[str, str]:
        number_strs_and_end_char_indices = [(mo.group(), mo.end()) for mo in NUMBER_PATTERN.finditer(text)]
        morphemes = self.morphological_analyzer.segment_text_into_morphemes(text)
        morpheme_index2char_index = self.morphological_analyzer.get_morpheme_index2char_index(text, morphemes)

        target_word2replacement = {}
        for number_str, end_char_index in number_strs_and_end_char_indices:
            end_morpheme = morphemes[morpheme_index2char_index.index(end_char_index)]
            if all(number_unit not in number_str for number_unit in NUMBER_UNITS) and end_morpheme[0] not in DATE_UNITS:
                target_word2replacement[number_str + end_morpheme] = (
                    self.get_perturbed_number(number_str) + end_morpheme
                )
        return target_word2replacement

    @staticmethod
    def get_perturbed_number(number_str: str) -> str:
        number = int(number_str)
        exponent = len(number_str) - 1
        number_candidates = [str(n) for n in range(10**exponent, 10 ** (exponent + 1)) if n != number]
        return random.choice(number_candidates)
