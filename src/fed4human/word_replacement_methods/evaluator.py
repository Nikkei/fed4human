import torch
from transformers import AutoTokenizer

from fed4human.constants import SYSTEM_PROMPT
from fed4human.utils import vllm_is_available


class Evaluator:
    def __init__(self, pretrained_llm_name_or_path: str) -> None:
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
            self.sampling_params = SamplingParams(max_tokens=self.max_tokens, temperature=0.0)
        else:
            from transformers import AutoModelForCausalLM

            self.llm_model = AutoModelForCausalLM.from_pretrained(
                pretrained_llm_name_or_path,
                dtype=dtype,
                device_map="auto",
                attn_implementation="sdpa",  # flash_attention_2
            )
            self.llm_model.eval()

    @staticmethod
    def get_messages4synonym_judgement(text: str, target_word: str, replacement: str) -> list[dict[str, str]]:
        text = [line for line in text.split("。") if target_word in line][0]
        user_prompt = (
            f"以下の文章中の「{target_word}」は「{replacement}」と同義語であるかを判定してください。\n"
            '回答は、同義語であれば{"answer": true}、そうでなければ{"answer": false}というJSON形式で出力してください。\n'
            f"文章: {text}"
        )
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]
        return messages

    def judge_synonym(self, synthetic_examples: list[dict[str, str]]) -> list[str]:
        batch_prompts = [
            self.llm_tokenizer.apply_chat_template(
                self.get_messages4synonym_judgement(e["text"], e["target_word"], e["replacement"]),
                add_generation_prompt=True,
                tokenize=False,
            )
            for e in synthetic_examples
        ]
        if self.vllm_is_available is True:
            batch_outputs = self.llm_model.generate(batch_prompts, self.sampling_params, use_tqdm=True)
            batch_completions = [o.outputs[0].text for o in batch_outputs]
        else:
            batch_inputs = self.llm_tokenizer(
                batch_prompts,
                padding=True,
                return_tensors="pt",
            ).to(self.llm_model.device)
            # generate function is already decorated with torch.no_grad
            batch_output_ids = self.llm_model.generate(
                **batch_inputs,
                max_new_tokens=self.max_tokens,
                do_sample=False,
            )
            batch_completion_ids = batch_output_ids[:, batch_inputs.input_ids.size(1) :]
            batch_completions = [
                self.llm_tokenizer.decode(completion_ids, skip_special_tokens=True)
                for completion_ids in batch_completion_ids
            ]
        return batch_completions
