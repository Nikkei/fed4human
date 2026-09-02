import random
from datetime import datetime

from fed4human.word_replacement_methods.base import BaseWordReplacementMethod
from fed4human.word_replacement_methods.constants import DATE_UNITS, NUMBER_PATTERN, DateUnit


class DateWordReplacementMethod(BaseWordReplacementMethod):
    def get_target_word2replacement(self, text: str, publish_date: datetime | None = None) -> dict[str, str]:
        number_strs_and_end_char_indices = [(mo.group(), mo.end()) for mo in NUMBER_PATTERN.finditer(text)]

        target_word2replacement = {}
        for number_str, end_char_index in number_strs_and_end_char_indices:
            if text[end_char_index] in DATE_UNITS and number_str.isdigit():
                date_unit = text[end_char_index]
                if len(number_str) == 4 and date_unit == DateUnit.YEAR.value:
                    publish_year = publish_date.year if publish_date else None
                    perturbed_year = self.get_perturbed_year(number_str, publish_year)
                    target_word2replacement[number_str + date_unit] = perturbed_year + date_unit
                elif len(number_str) <= 2 and date_unit == DateUnit.MONTH.value:
                    publish_month = publish_date.month if publish_date else None
                    if perturbed_month := self.get_perturbed_month(number_str, publish_month):
                        target_word2replacement[number_str + date_unit] = perturbed_month + date_unit
                elif len(number_str) <= 2 and date_unit == DateUnit.DAY.value:
                    publish_day = publish_date.day if publish_date else None
                    if perturbed_day := self.get_perturbed_day(number_str, publish_day):
                        target_word2replacement[number_str + date_unit] = perturbed_day + date_unit
        return target_word2replacement

    @staticmethod
    def get_perturbed_year(number_str: str, publish_year: int | None = None) -> str:
        number = int(number_str)
        range_min, range_max = -5, 5
        if publish_year:
            if number <= publish_year:
                range_max = -1
            else:
                range_min = 1
        delta = random.choice([n for n in range(range_min, range_max + 1) if n != 0])
        return str(number + delta)

    @staticmethod
    def get_perturbed_month(number_str: str, publish_month: int | None = None) -> str | None:
        number = int(number_str)
        range_min, range_max = 1, 12
        if publish_month:
            if number <= publish_month:
                range_max = publish_month
            else:
                range_min = publish_month
        number_candidates = [str(n) for n in range(range_min, range_max + 1) if n != number]
        return random.choice(number_candidates) if len(number_candidates) > 0 else None

    @staticmethod
    def get_perturbed_day(number_str: str, publish_day: int | None = None) -> str | None:
        number = int(number_str)
        range_min, range_max = 1, 31
        if publish_day:
            if number <= publish_day:
                range_max = publish_day
            else:
                range_min = publish_day
        number_candidates = [str(n) for n in range(range_min, range_max) if n != number]
        return random.choice(number_candidates) if len(number_candidates) > 0 else None
