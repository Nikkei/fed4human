import re
from dataclasses import dataclass

import spacy
from spacy.lang.ja import Japanese
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from fed4human.word_replacement_methods.base import BaseWordReplacementMethod
from fed4human.word_replacement_methods.constants import TARGET_NE_CATEGORIES


@dataclass
class NamedEntity:
    text: str
    start: int
    category: str


class NamedEntityWordReplacementMethod(BaseWordReplacementMethod):
    def __init__(self, pretrained_t5_name_or_path: str, ner_model_name: str) -> None:
        spacy.prefer_gpu()
        self.nlp: Japanese = spacy.load(ner_model_name)
        self.t5_tokenizer = AutoTokenizer.from_pretrained(pretrained_t5_name_or_path)
        self.t5_model = AutoModelForSeq2SeqLM.from_pretrained(pretrained_t5_name_or_path, device_map="auto")
        self.t5_model.eval()
        self.max_tokens = 16

        self.number_pattern = re.compile(r"\d+")

    def get_target_word2replacement(self, text: str) -> dict[str, dict[str, str]]:
        named_entities = [ne for ne in self.extract_named_entities(text) if ne.category in TARGET_NE_CATEGORIES]

        target_word2replacement = {}
        for named_entity in named_entities:
            # generate function is already decorated with torch.no_grad
            replacement_candidates = self.generate_replacement_candidates_using_t5(text, named_entity.text)
            if not replacement_candidates:
                continue
            replacements = self.filter_replacement_candidates(
                replacement_candidates,
                text,
                named_entity.text,
                named_entity.category,
            )
            if len(replacements) > 0:
                target_word2replacement[named_entity.text] = {
                    "replacement": replacements[0],
                    "subcategory": named_entity.category,
                }
        return target_word2replacement

    def extract_named_entities(self, text: str) -> list[NamedEntity]:
        doc = self.nlp(text)

        named_entities = []
        appeared = set()
        for ent in doc.ents:
            if self.number_pattern.search(ent.text) is None and ent.text not in appeared:
                named_entities.append(NamedEntity(text=ent.text, start=int(ent.start_char), category=ent.label_))
                appeared.add(ent.text)
        return named_entities

    def generate_replacement_candidates_using_t5(
        self, text: str, target_word: str, top_k: int = 8, max_length: int = 512
    ) -> list[str] | None:
        sentinel_token1 = self.t5_tokenizer.additional_special_tokens[0]
        sentinel_id1 = self.t5_tokenizer.convert_tokens_to_ids(sentinel_token1)
        masked_text = text.replace(target_word, sentinel_token1)
        input_ids = self.t5_tokenizer.encode(
            masked_text,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        ).to(self.t5_model.device)
        _, seq_len = input_ids.shape
        if sentinel_id1 not in input_ids:
            return None
        # sample top_k candidates of a first token
        decoder_input_ids = self.t5_model.generate(
            input_ids,
            num_return_sequences=top_k,
            num_beams=top_k,
            max_length=3,  # <pad> <extra_id_0> <candidate_of_a_first_token>
        )

        sentinel_token2 = self.t5_tokenizer.additional_special_tokens[1]
        # greedy decoding from the sampled first tokens
        outputs = self.t5_model.generate(
            input_ids.expand(top_k, seq_len),
            decoder_input_ids=decoder_input_ids,
            eos_token_id=self.t5_tokenizer.vocab[sentinel_token2],
            max_length=self.max_tokens,
        )

        replacement_candidates = []
        for output in outputs:
            replacement_candidate = self.t5_tokenizer.decode(output, skip_special_tokens=True)
            if replacement_candidate != target_word:
                replacement_candidates.append(replacement_candidate)
        return replacement_candidates

    def filter_replacement_candidates(
        self,
        replacement_candidates: list[str],
        text: str,
        target_word: str,
        category: str,
    ) -> list[str]:
        replacements = []
        for replacement_candidate in replacement_candidates:
            if (
                replacement_candidate in text
                or replacement_candidate in target_word
                or target_word in replacement_candidate
                or "[UNK]" in replacement_candidate
            ):
                continue
            replaced_text = text.replace(target_word, replacement_candidate)
            named_entities = self.extract_named_entities(replaced_text)
            for named_entity in named_entities:
                # retain it as a replacement if it belongs to the same category as the target word
                if named_entity.text == replacement_candidate and named_entity.category == category:
                    replacements.append(replacement_candidate)
        return replacements
