import re
from collections import defaultdict
from dataclasses import dataclass
from os.path import commonprefix

from fed4human.deep_learning.constants import END_TOKEN, NO_ANSWER_ASSISTANT_PROMPT, SPLIT_PAT, START_TOKEN


@dataclass
class Outcomes:
    TP: int = 0
    FP: int = 0
    FN: int = 0

    def compute_precision(self) -> float:
        return self.TP / (self.TP + self.FP) if (self.TP + self.FP) >= 1 else 0.0

    def compute_recall(self) -> float:
        return self.TP / (self.TP + self.FN) if (self.TP + self.FN) >= 1 else 0.0

    def compute_f1_score(self) -> float:
        prec = self.compute_precision()
        rec = self.compute_recall()
        return 2 * prec * rec / (prec + rec) if (prec + rec) > 0.0 else 0.0


class Metric:
    def __init__(self):
        self.itemize_pat = re.compile(r"^- (?P<item>.+)$", re.MULTILINE)

        self.category2sentence_level_outcomes = defaultdict(Outcomes)
        self.category2difficulty2sentence_level_outcomes = defaultdict(lambda: defaultdict(Outcomes))

        self.enclosed_pat = re.compile(rf"{re.escape(START_TOKEN)}(?P<word>.+?){re.escape(END_TOKEN)}")

        self.category2word_level_outcomes = defaultdict(Outcomes)
        self.category2difficulty2word_level_outcomes = defaultdict(lambda: defaultdict(Outcomes))

    def update_sentence_level_outcomes(self, in_obj: dict[str, object]) -> None:
        # "<think>\n\n</think>\n\n..."" -> "..."
        completion = in_obj["completion"].split("</think>\n\n")[-1]

        preds = {
            s.replace(START_TOKEN, "").replace(END_TOKEN, "").strip()
            for mo in self.itemize_pat.finditer(completion)
            for s in SPLIT_PAT.split(mo.group("item"))
        }
        if len(preds) == 0:
            preds = {NO_ANSWER_ASSISTANT_PROMPT}

        if in_obj["category"] == "negative":
            golds = {NO_ANSWER_ASSISTANT_PROMPT}
        else:
            golds = {
                f'{s.replace(in_obj["target_word"], in_obj["replacement"])}'.strip()
                for s in SPLIT_PAT.split(in_obj["text"])
                if in_obj["target_word"] in s
            }

        for pred in preds:
            if pred in golds:
                self.category2sentence_level_outcomes[in_obj["category"]].TP += 1
                self.category2sentence_level_outcomes["micro"].TP += 1
                self.category2difficulty2sentence_level_outcomes[in_obj["category"]][in_obj["difficulty"]].TP += 1
                self.category2difficulty2sentence_level_outcomes["micro"][in_obj["difficulty"]].TP += 1
                golds.remove(pred)
            else:
                self.category2sentence_level_outcomes[in_obj["category"]].FP += 1
                self.category2sentence_level_outcomes["micro"].FP += 1
                self.category2difficulty2sentence_level_outcomes[in_obj["category"]][in_obj["difficulty"]].FP += 1
                self.category2difficulty2sentence_level_outcomes["micro"][in_obj["difficulty"]].FP += 1
        self.category2sentence_level_outcomes[in_obj["category"]].FN += len(golds)
        self.category2sentence_level_outcomes["micro"].FN += len(golds)
        self.category2difficulty2sentence_level_outcomes[in_obj["category"]][in_obj["difficulty"]].FN += len(golds)
        self.category2difficulty2sentence_level_outcomes["micro"][in_obj["difficulty"]].FN += len(golds)

    def update_word_level_outcomes(self, in_obj: dict[str, object]) -> None:
        # "<think>\n\n</think>\n\n..."" -> "..."
        completion = in_obj["completion"].split("</think>\n\n")[-1]

        preds = {
            (s.replace(START_TOKEN, "").replace(END_TOKEN, "").strip(), word_mo.group("word"))
            for item_mo in self.itemize_pat.finditer(completion)
            for s in SPLIT_PAT.split(item_mo.group("item"))
            for word_mo in self.enclosed_pat.finditer(s)
        }
        golds = {
            f'{s.replace(in_obj["target_word"], in_obj["replacement"])}'.strip()
            for s in SPLIT_PAT.split(in_obj["text"])
            if in_obj["target_word"] in s
        }
        if in_obj["category"] == "negative":
            if len(preds) == 0:
                self.category2word_level_outcomes[in_obj["category"]].TP += 1
                self.category2word_level_outcomes["micro"].TP += 1
                self.category2difficulty2word_level_outcomes[in_obj["category"]][in_obj["difficulty"]].TP += 1
                self.category2difficulty2word_level_outcomes["micro"][in_obj["difficulty"]].TP += 1
            else:
                self.category2word_level_outcomes[in_obj["category"]].FP += len(preds)
                self.category2word_level_outcomes["micro"].FP += len(preds)
                self.category2word_level_outcomes[in_obj["category"]].FN += 1
                self.category2word_level_outcomes["micro"].FN += 1
                self.category2difficulty2word_level_outcomes[in_obj["category"]][in_obj["difficulty"]].FP += len(preds)
                self.category2difficulty2word_level_outcomes["micro"][in_obj["difficulty"]].FP += len(preds)
                self.category2difficulty2word_level_outcomes[in_obj["category"]][in_obj["difficulty"]].FN += 1
                self.category2difficulty2word_level_outcomes["micro"][in_obj["difficulty"]].FN += 1
        else:
            num_golds = in_obj["text"].count(in_obj["target_word"])
            prefix = commonprefix([in_obj["target_word"], in_obj["replacement"]])
            suffix = commonprefix([in_obj["target_word"][::-1], in_obj["replacement"][::-1]])
            minimal_edit_span = in_obj["replacement"][len(prefix) : len(in_obj["replacement"]) - len(suffix)]
            if minimal_edit_span == "":
                minimal_edit_span = in_obj["replacement"]
            for pred, pred_word in preds:
                if pred in golds and minimal_edit_span in pred_word:
                    self.category2word_level_outcomes[in_obj["category"]].TP += 1
                    self.category2word_level_outcomes["micro"].TP += 1
                    self.category2difficulty2word_level_outcomes[in_obj["category"]][in_obj["difficulty"]].TP += 1
                    self.category2difficulty2word_level_outcomes["micro"][in_obj["difficulty"]].TP += 1
                    num_golds -= 1
                else:
                    self.category2word_level_outcomes[in_obj["category"]].FP += 1
                    self.category2word_level_outcomes["micro"].FP += 1
                    self.category2difficulty2word_level_outcomes[in_obj["category"]][in_obj["difficulty"]].FP += 1
                    self.category2difficulty2word_level_outcomes["micro"][in_obj["difficulty"]].FP += 1
            self.category2word_level_outcomes[in_obj["category"]].FN += num_golds
            self.category2word_level_outcomes["micro"].FN += num_golds
            self.category2difficulty2word_level_outcomes[in_obj["category"]][in_obj["difficulty"]].FN += num_golds
            self.category2difficulty2word_level_outcomes["micro"][in_obj["difficulty"]].FN += num_golds
