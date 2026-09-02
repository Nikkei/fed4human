import json

import pandas as pd
import torch
from gensim.models.keyedvectors import KeyedVectors
from transformers import AutoTokenizer

from fed4human.constants import SYSTEM_PROMPT
from fed4human.utils import vllm_is_available
from fed4human.word_replacement_methods.base import BaseWordReplacementMethod
from fed4human.word_replacement_methods.morphological_analyzer import MorphologicalAnalyzer
from fed4human.word_replacement_methods.utils import get_similar_words_using_word2vec


class AntonymWordReplacementMethod(BaseWordReplacementMethod):
    def __init__(
        self,
        w2v_model_path: str,
        pretrained_llm_name_or_path: str,
        top_k: int = 5,
        antonym_dict_path: str | None = None,
    ) -> None:
        self.morphological_analyzer = MorphologicalAnalyzer()
        self.w2v_model = KeyedVectors.load_word2vec_format(w2v_model_path, binary=False)
        self.llm_tokenizer = AutoTokenizer.from_pretrained(pretrained_llm_name_or_path)
        cc_major, _ = torch.cuda.get_device_capability()
        dtype = "bfloat16" if cc_major >= 8 else "float16"
        self.max_tokens = 64
        self.vllm_is_available = vllm_is_available()
        if self.vllm_is_available is True:
            from vllm import LLM, SamplingParams

            self.llm_model = LLM(
                model=pretrained_llm_name_or_path,
                dtype=dtype,
                tensor_parallel_size=torch.cuda.device_count(),
            )
            self.sampling_params = SamplingParams(max_tokens=self.max_tokens, top_p=0.95, temperature=0.0)
        else:
            from transformers import AutoModelForCausalLM

            self.llm_model = AutoModelForCausalLM.from_pretrained(
                pretrained_llm_name_or_path,
                dtype=dtype,
                device_map="auto",
                attn_implementation="sdpa",  # flash_attention_2
            )
            self.llm_model.eval()
        self.top_k = top_k

        if antonym_dict_path:
            df = pd.read_csv(antonym_dict_path)
            self.antonym_dict = dict(zip(df["target_word"], df["replacement"]))
        else:
            self.antonym_dict = {}

    def get_target_word2replacement(self, text: str) -> dict[str, str]:
        morpheme2feature_dict = self.morphological_analyzer.get_morpheme2feature_dict(text)

        target_word2replacement = {}
        for morpheme, feature_dict in morpheme2feature_dict.items():
            if morpheme in self.antonym_dict.keys():
                target_word2replacement[morpheme] = self.antonym_dict[morpheme]
            elif feature_dict.pos == "名詞" and "サ変可能" in feature_dict.subpos:
                replacement_canidates = [
                    similar_word
                    for similar_word in get_similar_words_using_word2vec(morpheme, self.w2v_model)
                    if morpheme not in similar_word
                ][: self.top_k]
                resp = self.generate_antonym(morpheme)
                if resp["answer"] != "なし" and resp["answer"] in replacement_canidates:
                    target_word2replacement[morpheme] = resp["answer"]
        self.antonym_dict.update(target_word2replacement)  # cache
        return target_word2replacement

    def generate_antonym(self, target_word: str) -> dict[str, str]:
        user_prompt = (
            f"「{target_word}」の対義語として最も適切な単語を1つ答えてください。\n"
            '回答は{"answer": 対義語}というJSON形式で出力し、適切な単語がない場合は{"answer": "なし"}と答えてください。'
        )
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]
        if self.vllm_is_available is True:
            prompts = self.llm_tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=False,
            )
            outputs = self.llm_model.generate(prompts, self.sampling_params, use_tqdm=False)  # list[RequestOutput]
            completion = outputs[0].outputs[0].text
        else:
            inputs = self.llm_tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_tensors="pt",
            ).to(self.llm_model.device)
            # generate function is already decorated with torch.no_grad
            outputs = self.llm_model.generate(
                **inputs,
                max_new_tokens=self.max_tokens,
                do_sample=False,
                pad_token_id=self.llm_tokenizer.pad_token_id,
            )
            completion_ids = outputs[0, inputs.input_ids.size(1) :]
            completion = self.llm_tokenizer.decode(completion_ids, skip_special_tokens=True)
        try:
            obj = json.loads(completion)
            return obj
        except json.decoder.JSONDecodeError:
            return {"answer": "なし"}
