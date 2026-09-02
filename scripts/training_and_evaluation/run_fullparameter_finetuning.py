import sys
from math import ceil
from pprint import pprint

import hydra
from omegaconf import DictConfig, OmegaConf
from trl import SFTTrainer

from fed4human.deep_learning.data_modules import preprocess_examples
from fed4human.deep_learning.utils import set_omegaconf_resolver
from fed4human.utils import set_seed


@hydra.main(config_path="../../config/training")
def main(config: DictConfig):
    OmegaConf.resolve(config)
    set_seed(config.seed)
    pprint(OmegaConf.to_container(config))

    tokenizer = hydra.utils.instantiate(config.data_module.tokenizer)

    train_dataset = hydra.utils.instantiate(config.data_module.dataset)
    train_dataset = train_dataset.map(preprocess_examples, fn_kwargs={"tokenizer": tokenizer})
    # not to use `text` column internally
    train_dataset = train_dataset.select_columns(["prompt", "completion"])
    print(f'sample prompt: {train_dataset[0]["prompt"]}')
    print(f'sample completion: {train_dataset[0]["completion"]}')

    model = hydra.utils.instantiate(config.model, device_map=None)  # must set device_map to None for DeepSpeed
    model.config.use_cache = False  # to save GPU VRAM for generation

    max_steps = min(ceil(len(train_dataset) / config.effective_batch_size), 1024)
    training_arguments = hydra.utils.instantiate(
        config.training_arguments,
        max_steps=max_steps,
        deepspeed=config.deepspeed,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_arguments,
        train_dataset=train_dataset,
        processing_class=tokenizer,
    )

    trainer.train()


if __name__ == "__main__":
    # processes are spawned with --local_rank= argument, which Hydra doesn't recognize
    sys.argv = [arg for arg in sys.argv if arg.startswith("--local_rank=") is False]
    set_omegaconf_resolver()
    main()
