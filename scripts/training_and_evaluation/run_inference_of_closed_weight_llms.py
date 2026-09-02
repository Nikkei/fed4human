import json
from pprint import pprint

import hydra
from dotenv import load_dotenv
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from fed4human.closed_weight_llm_evaluators.utils import get_evaluator
from fed4human.deep_learning.data_modules import get_prompts4closed_weight_llms
from fed4human.deep_learning.utils import set_omegaconf_resolver
from fed4human.utils import set_seed


@hydra.main(config_path="../../config/evaluation")
def main(config: DictConfig):
    load_dotenv()

    OmegaConf.resolve(config)
    set_seed(config.seed)
    pprint(OmegaConf.to_container(config))

    if config.num_shot >= 1:
        with config.data_module.few_shot_examples_file.open(mode="r") as fin:
            few_shot_examples = [json.loads(line) for line in fin][: config.num_shot]
    else:
        few_shot_examples = None  # zero-shot

    eval_dataset = hydra.utils.instantiate(config.data_module.dataset)
    prompts, _ = get_prompts4closed_weight_llms(eval_dataset[0], few_shot_examples)
    print("** sample prompts **")
    for i, prompt in enumerate(prompts):
        role = "user" if i % 2 == 0 else "assistant"
        print(f"** {role} **\n{prompt}")

    evaluator = get_evaluator(config.model_id)

    if config.out_file.exists() is True:
        with config.out_file.open(mode="r") as fin:
            resume_index = sum(1 for _ in fin)
    else:
        resume_index = 0
    bar = tqdm(eval_dataset, desc="for loop of eval_data_loader")
    with config.out_file.open(mode="a") as fout:
        for i, example in enumerate(bar):
            if i < resume_index:
                continue
            prompts, _ = get_prompts4closed_weight_llms(example, few_shot_examples)
            messages = evaluator.get_messages(prompts)
            output = evaluator.get_output(messages)
            try:
                difficulty = example["difficulty"]
            except KeyError:
                difficulty = -100
            out_obj = {
                "completion": output["completion"],
                "category": example["category"],
                "difficulty": difficulty,
                "text": example["text"],
                "target_word": example["target_word"],
                "replacement": example["replacement"],
            }
            fout.write(json.dumps(out_obj, ensure_ascii=False) + "\n")
            bar.set_postfix(cost=f"${evaluator.compute_cost()}")


if __name__ == "__main__":
    set_omegaconf_resolver()
    main()
