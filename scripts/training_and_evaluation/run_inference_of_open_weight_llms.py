import json
from pprint import pprint

import hydra
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm

from fed4human.deep_learning.data_modules import preprocess_examples
from fed4human.deep_learning.utils import set_omegaconf_resolver
from fed4human.utils import set_seed


@hydra.main(config_path="../../config/evaluation")
def main(config: DictConfig):
    OmegaConf.resolve(config)
    set_seed(config.seed)
    pprint(OmegaConf.to_container(config))

    tokenizer = hydra.utils.instantiate(config.data_module.tokenizer)

    if config.setting == "fewshot":
        with config.data_module.few_shot_examples_file.open(mode="r") as fin:
            few_shot_examples = [json.loads(line) for line in fin]
    else:
        few_shot_examples = None

    eval_dataset = hydra.utils.instantiate(config.data_module.dataset)
    eval_dataset = eval_dataset.map(
        preprocess_examples,
        fn_kwargs={"tokenizer": tokenizer, "few_shot_examples": few_shot_examples},
    )
    # batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'datetime.datetime'>
    eval_dataset = eval_dataset.remove_columns(["datetime"])
    eval_data_loader = DataLoader(
        eval_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        pin_memory=True,
    )

    model = hydra.utils.instantiate(config.model.transformers)
    model.eval()

    bar = tqdm(eval_data_loader, desc="for loop of eval_data_loader")
    config.out_file.parent.mkdir(parents=True, exist_ok=True)
    with config.out_file.open(mode="w") as fout:
        for batch in bar:
            batch_inputs = tokenizer(batch["prompt"], padding=True, return_tensors="pt").to(model.device)
            batch_output_ids = model.generate(
                **batch_inputs,
                max_new_tokens=config.max_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
            batch_completion_ids = batch_output_ids[:, batch_inputs.input_ids.size(1) :]
            batch_completions = [
                tokenizer.decode(completion_ids, skip_special_tokens=True) for completion_ids in batch_completion_ids
            ]
            for i, completion in enumerate(batch_completions):
                try:
                    difficulty = batch["difficulty"][i]
                except KeyError:
                    difficulty = -100
                out_obj = {
                    "completion": completion,
                    "category": batch["category"][i],
                    "difficulty": difficulty,
                    "text": batch["text"][i],
                    "target_word": batch["target_word"][i],
                    "replacement": batch["replacement"][i],
                }
                fout.write(json.dumps(out_obj, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    set_omegaconf_resolver()
    main()
