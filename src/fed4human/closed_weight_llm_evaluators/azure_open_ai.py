from os import environ
from time import sleep

from openai import AzureOpenAI
from openai.types.responses import ResponseOutputMessage
from tenacity import retry, stop_after_attempt, wait_exponential

from fed4human.closed_weight_llm_evaluators.base import BaseEvaluator, retry_error_callback
from fed4human.closed_weight_llm_evaluators.constants import AZURE_OPEN_AI_MODEL_ID2PRICING


class AzureOpenAIEvaluator(BaseEvaluator):
    def __init__(self, model_id: str, max_tokens: int = 128) -> None:
        super().__init__(model_id, max_tokens=max_tokens)
        self.client = AzureOpenAI(
            azure_endpoint=environ["AZURE_ENDPOINT"],
            azure_deployment=model_id,
            # cf. https://learn.microsoft.com/en-us/azure/ai-foundry/openai/api-version-lifecycle?view=foundry-classic&tabs=python#changes-between-v1-preview-release-and-2025-04-01-preview
            api_version=environ["API_VERSION"],
            api_key=environ["API_KEY"],
            timeout=16,
        )

    def get_messages(self, prompts: list[str]) -> list[dict[str, object]]:
        messages = [{"role": "system", "content": self.system_prompt}]
        for i, prompt in enumerate(prompts):
            role = "user" if i % 2 == 0 else "assistant"
            messages.append({"role": role, "content": prompt})
        return messages

    @retry(
        stop=stop_after_attempt(4),
        wait=wait_exponential(multiplier=2, max=8),
        retry_error_callback=retry_error_callback,
    )
    def get_output(self, messages: list[dict[str, object]]) -> dict[str, str]:
        resp = self.client.responses.create(
            model=self.model_id,
            max_output_tokens=self.max_tokens,
            input=messages,
            temperature=0.0,
        )
        sleep(self.delay)

        self.num_input_tokens += resp.usage.input_tokens
        self.num_output_tokens += resp.usage.output_tokens

        response_output_messages = [o for o in resp.output if isinstance(o, ResponseOutputMessage)]
        completion = response_output_messages[0].content[0].text if len(response_output_messages) >= 1 else ""
        return {"completion": completion}

    def compute_cost(self) -> float:
        pricing = AZURE_OPEN_AI_MODEL_ID2PRICING[self.model_id]
        price = (
            pricing["price_per_input_token"] * self.num_input_tokens
            + pricing["price_per_output_token"] * self.num_output_tokens
        )
        return round(price, 3)
