import random
import re

from fed4human.word_replacement_methods.base import BaseWordReplacementMethod
from fed4human.word_replacement_methods.constants import DATE_UNITS, NUMBER_PATTERN, NUMBER_UNITS, NumberUnit
from fed4human.word_replacement_methods.morphological_analyzer import MorphologicalAnalyzer


class DigitWordReplacementMethod(BaseWordReplacementMethod):
    def __init__(self) -> None:
        self.morphological_analyzer = MorphologicalAnalyzer()

        self.two_zeros_pattern = re.compile(r"[0-9]+0{2,}")

    def get_target_word2replacement(self, text: str) -> dict[str, str]:
        number_strs_and_end_char_indices = [(mo.group(), mo.end()) for mo in NUMBER_PATTERN.finditer(text)]
        morphemes = self.morphological_analyzer.segment_text_into_morphemes(text)
        morpheme_index2char_index = self.morphological_analyzer.get_morpheme_index2char_index(text, morphemes)

        target_word2replacement = {}
        for number_str, end_char_index in number_strs_and_end_char_indices:
            number_unit_count = sum([number_str.count(number_unit) for number_unit in NUMBER_UNITS])
            end_morpheme = morphemes[morpheme_index2char_index.index(end_char_index)]
            if self.two_zeros_pattern.match(number_str) and end_morpheme[0] not in DATE_UNITS:
                target_word2replacement[number_str + end_morpheme] = number_str[:-1] + end_morpheme
            elif number_unit_count >= 1 and number_str[-1] in NUMBER_UNITS and end_morpheme[0] not in DATE_UNITS:
                target_word2replacement[number_str + end_morpheme] = (
                    self.get_perturbed_number(number_str) + end_morpheme
                )
        return target_word2replacement

    @staticmethod
    def get_perturbed_number(number_str: str) -> str:
        switch = random.randint(0, 1)
        if switch == 1 or NumberUnit.CHO.value in number_str:
            table = str.maketrans(
                {NumberUnit.CHO.value: NumberUnit.OKU.value, NumberUnit.OKU.value: NumberUnit.MAN.value}
            )
            if NumberUnit.MAN.value in number_str:
                order = 0
                for char in number_str[: number_str.index(NumberUnit.MAN.value) + 1][::-1]:
                    if char in NUMBER_UNITS:
                        break
                    else:
                        order += 1
                number_str = number_str.replace(NumberUnit.MAN.value, "0" * (order - 2))
        else:
            table = str.maketrans(
                {NumberUnit.OKU.value: NumberUnit.CHO.value, NumberUnit.MAN.value: NumberUnit.OKU.value}
            )
        return number_str.translate(table)
