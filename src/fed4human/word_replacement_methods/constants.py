import re
from enum import Enum


class Category(Enum):
    NAMED_ENTITY = "named_entity"
    KANJI_MISCONVERSION = "kanji_misconversion"
    ANTONYM = "antonym"
    NUMERICAL_VALUE = "numerical_value"
    DATE = "date"
    DIGIT = "digit"
    UNIT = "unit"


class DateUnit(Enum):
    YEAR = "年"
    MONTH = "月"
    DAY = "日"


class NumberUnit(Enum):
    CHO = "兆"
    OKU = "億"
    MAN = "万"


TARGET_NE_CATEGORIES = {
    "Person",
    "Government",
    "Position_Vocation",
    "Continental_Region",
    "Country",
}
DATE_UNITS = [date_unit.value for date_unit in DateUnit]
NUMBER_UNITS = [number_unit.value for number_unit in NumberUnit]
NUMBER_PATTERN = re.compile(rf'\d+?[\d{"".join(NUMBER_UNITS)}]+')
