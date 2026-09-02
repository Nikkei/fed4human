from abc import ABC, abstractmethod

from tenacity import RetryCallState

from fed4human.constants import SYSTEM_PROMPT


def retry_error_callback(retry_call_state: RetryCallState) -> str:
    e = retry_call_state.outcome.exception()
    return {
        "completion": "RuntimeError",
        "debug": f"{e.__class__.__name__}: {e}",
    }


class BaseEvaluator(ABC):
    def __init__(self, model_id: str, max_tokens: int = 128) -> None:
        self.model_id = model_id

        self.system_prompt = SYSTEM_PROMPT

        self.max_tokens = max_tokens
        self.delay = 1.0

        self.num_input_tokens = 0
        self.num_output_tokens = 0

    @abstractmethod
    def get_messages(self, prompts: list[str]) -> list[dict[str, object]]:
        raise NotImplementedError

    @abstractmethod
    def get_output(self, messages: list[dict[str, object]]) -> str:
        raise NotImplementedError

    @abstractmethod
    def compute_cost(self) -> float:
        raise NotImplementedError
