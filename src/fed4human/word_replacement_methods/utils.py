from argparse import ArgumentTypeError
from datetime import datetime, timezone

from gensim.models.keyedvectors import KeyedVectors

from fed4human.word_replacement_methods.base import BaseWordReplacementMethod
from fed4human.word_replacement_methods.constants import Category


def get_similar_words_using_word2vec(
    target_word: str,
    w2v_model: KeyedVectors,
    topn: int = 10,
    threshold: float = 0.0,
) -> list[str]:
    try:
        topn_similar_words = w2v_model.most_similar(positive=[target_word], topn=topn)
        similar_words = []
        for word, score in topn_similar_words:
            if score <= threshold:
                continue
            preprocessed = word.split("_")[0].replace("[", "").replace("]", "")
            if preprocessed != target_word:
                similar_words.append(preprocessed)
        return similar_words
    except KeyError:
        return []


def category_type(value: str) -> Category:
    try:
        return Category(value)
    except ValueError:
        raise ArgumentTypeError(f"invalid category ({value})")


def until_type(value: str) -> datetime:
    try:
        return datetime.strptime(value, "%Y-%m-%d").astimezone(timezone.utc)
    except ValueError:
        raise ArgumentTypeError(f"invalid until ({value})")


def get_word_replacement_method(category: Category, config: dict[str, str]) -> BaseWordReplacementMethod:
    if category == Category.NAMED_ENTITY:
        # lazy loading
        from fed4human.word_replacement_methods.named_entity import NamedEntityWordReplacementMethod

        return NamedEntityWordReplacementMethod(
            pretrained_t5_name_or_path=config["pretrained_t5_name_or_path"],
            ner_model_name=config["ner_model_name"],
        )
    elif category == Category.KANJI_MISCONVERSION:
        from fed4human.word_replacement_methods.kanji_misconversion import KanjiMisconversionWordReplacementMethod

        return KanjiMisconversionWordReplacementMethod(ner_model_name=config["ner_model_name"])
    elif category == Category.ANTONYM:
        from fed4human.word_replacement_methods.antonym import AntonymWordReplacementMethod

        return AntonymWordReplacementMethod(
            w2v_model_path=config["w2v_model_path"],
            pretrained_llm_name_or_path=config["pretrained_llm_name_or_path"],
            top_k=5,
            antonym_dict_path=config["antonym_dict_path"],
        )
    elif category == Category.NUMERICAL_VALUE:
        from fed4human.word_replacement_methods.numerical_value import NumericalValueWordReplacementMethod

        return NumericalValueWordReplacementMethod()
    elif category == Category.DATE:
        from fed4human.word_replacement_methods.date import DateWordReplacementMethod

        return DateWordReplacementMethod()
    elif category == Category.DIGIT:
        from fed4human.word_replacement_methods.digit import DigitWordReplacementMethod

        return DigitWordReplacementMethod()
    elif category == Category.UNIT:
        from fed4human.word_replacement_methods.unit import UnitWordReplacementMethod

        return UnitWordReplacementMethod(w2v_model_path=config["w2v_model_path"])
    else:
        raise ValueError("unsupported category")
