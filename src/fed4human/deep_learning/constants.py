import re

SPLIT_PAT = re.compile(r"(?<=。)(?!$)")

START_TOKEN = "<|factual_error_start|>"
END_TOKEN = "<|factual_error_end|>"

NO_ANSWER_ASSISTANT_PROMPT = "事実と異なる箇所はありません。"

# INSTRUCTION = (
#     '以下の記事中に事実と異なる箇所があれば、該当箇所が含まれる文の番号をカンマ(",")区切りで列挙してください。'
#     f"記事中に事実と異なる箇所がなければ、「{NO_ANSWER_ASSISTANT_PROMPT}」と答えてください。"
# )
INSTRUCTION = (
    f"以下の記事中に事実と異なる箇所があれば、該当箇所を{START_TOKEN}{END_TOKEN}で囲み、それが含まれる文をそのまま抜き出してください。"
    "回答は「- 文1\n- 文2\n…」のように箇条書きで列挙してください。"
    f"事実と異なる箇所がなければ、「{NO_ANSWER_ASSISTANT_PROMPT}」と回答してください。"
)
