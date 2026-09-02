from abc import ABC, abstractmethod


class BaseWordReplacementMethod(ABC):
    @abstractmethod
    def get_target_word2replacement(self, text: str, **kwargs) -> list[dict[str, dict[str, str] | str]]:
        raise NotImplementedError
