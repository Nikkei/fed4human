from fed4human.closed_weight_llm_evaluators.azure_open_ai import AzureOpenAIEvaluator
from fed4human.closed_weight_llm_evaluators.base import BaseEvaluator
from fed4human.closed_weight_llm_evaluators.constants import AZURE_OPEN_AI_MODEL_ID2PRICING


def get_evaluator(model_id: str, max_tokens: int = 1024) -> BaseEvaluator:
    if model_id in AZURE_OPEN_AI_MODEL_ID2PRICING:
        evaluator = AzureOpenAIEvaluator(model_id, max_tokens=max_tokens)
    else:
        raise ValueError("unsupported model_id")
    return evaluator
