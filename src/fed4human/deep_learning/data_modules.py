from datetime import datetime

from transformers import PreTrainedTokenizerBase

from fed4human.constants import SYSTEM_PROMPT
from fed4human.deep_learning.constants import (
    END_TOKEN,
    INSTRUCTION,
    NO_ANSWER_ASSISTANT_PROMPT,
    SPLIT_PAT,
    START_TOKEN,
)


def get_publish_date(datetime_object_or_string: datetime | str) -> str:
    if isinstance(datetime_object_or_string, datetime):
        return datetime_object_or_string.strftime("%-Y年%-m月%-d日")
    else:
        return datetime.strptime(datetime_object_or_string, "%Y-%m-%dT%H:%M:%SZ").strftime("%-Y年%-m月%-d日")


def get_user_and_assistant_prompts(
    example: dict[str, str | datetime],
    include_instruction: bool = True,
) -> tuple[str, str]:
    publish_date = get_publish_date(example["datetime"])
    sentences = SPLIT_PAT.split(example["text"])
    if example["category"] == "negative":
        text = example["text"]
        # text = "\n".join(f"({i + 1}){s}" for i, s in enumerate(sentences))
        answers = [NO_ANSWER_ASSISTANT_PROMPT]
    else:
        text = example["text"].replace(example["target_word"], example["replacement"])
        answers = [
            f'- {s.replace(example["target_word"], START_TOKEN + example["replacement"] + END_TOKEN)}'
            for s in sentences
            if example["target_word"] in s
        ]
        # text = "\n".join(
        #     f'({i + 1}){s.replace(example["target_word"], example["replacement"])}' for i, s in enumerate(sentences)
        # )
        # answers = [str(i + 1) for i, s in enumerate(sentences) if example["target_word"] in s]
    user_prompt = INSTRUCTION + "\n\n" if include_instruction is True else ""
    user_prompt += f"記事（{publish_date}掲載）: {text}"
    assistant_prompt = "\n".join(answers)
    # assistant_prompt = ",".join(answers)
    return user_prompt, assistant_prompt


def preprocess_examples(
    example: dict[str, str | datetime],
    tokenizer: PreTrainedTokenizerBase,
    few_shot_examples: list[dict[str, str]] | None = None,
) -> dict[str, str]:
    prompt_messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    if few_shot_examples:
        for i, few_shot_example in enumerate(few_shot_examples):
            user_prompt, assistant_prompt = get_user_and_assistant_prompts(
                few_shot_example,
                include_instruction=i == 0,
            )
            prompt_messages.append({"role": "user", "content": user_prompt})
            prompt_messages.append({"role": "assistant", "content": assistant_prompt})
    user_prompt, assistant_prompt = get_user_and_assistant_prompts(
        example,
        include_instruction=few_shot_examples is None,
    )
    prompt_messages.append({"role": "user", "content": user_prompt})
    completion_messages = [{"role": "assistant", "content": assistant_prompt}]

    prompt = tokenizer.apply_chat_template(prompt_messages, add_generation_prompt=True, tokenize=False)
    text = tokenizer.apply_chat_template(
        prompt_messages + completion_messages, add_generation_prompt=False, tokenize=False
    )
    return {"prompt": prompt, "completion": text[len(prompt) :], "answers": assistant_prompt}


def get_user_and_assistant_prompts4closed_weight_llms(
    example: dict[str, str | datetime],
    include_instruction: bool = True,
) -> tuple[str, str]:
    publish_date = get_publish_date(example["datetime"])
    sentences = SPLIT_PAT.split(example["text"])
    if example["category"] == "negative":
        text = example["text"]
        # text = "\n".join(f"({i + 1}){s}" for i, s in enumerate(sentences))
        answers = [NO_ANSWER_ASSISTANT_PROMPT]
    else:
        text = example["text"].replace(example["target_word"], example["replacement"])
        answers = [
            f'- {s.replace(example["target_word"], START_TOKEN + example["replacement"] + END_TOKEN)}'
            for s in sentences
            if example["target_word"] in s
        ]
        # text = "\n".join(
        #     f'({i + 1}){s.replace(example["target_word"], example["replacement"])}' for i, s in enumerate(sentences)
        # )
        # answers = [str(i + 1) for i, s in enumerate(sentences) if example["target_word"] in s]
    user_prompt = INSTRUCTION + "\n\n" if include_instruction is True else ""
    user_prompt += f"記事（{publish_date}掲載）: {text}"
    assistant_prompt = "\n".join(answers)
    # assistant_prompt = ",".join(answers)
    return user_prompt, assistant_prompt


def get_prompts4closed_weight_llms(
    example: dict[str, str], few_shot_examples: list[dict[str, str]] | None = None
) -> tuple[list[str], str]:
    prompts = []
    if few_shot_examples:
        for i, few_shot_example in enumerate(few_shot_examples):
            user_prompt, assistant_prompt = get_user_and_assistant_prompts4closed_weight_llms(
                few_shot_example,
                include_instruction=i == 0,
            )
            prompts.append(user_prompt)
            prompts.append(assistant_prompt)
    user_prompt, assistant_prompt = get_user_and_assistant_prompts4closed_weight_llms(
        example,
        include_instruction=few_shot_examples is None,
    )
    prompts.append(user_prompt)
    return prompts, assistant_prompt
