import configparser

import os

import torch
from trl import SFTTrainer, SFTConfig

from data import get_dataset
from load import load, load_fine_tuning_model
from utils.logger import get_logger


config = configparser.ConfigParser()
config.read("config/config.ini")
logger=get_logger(config)



'''
CUDA报OOM错误
'''
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
'''
使用CUDA
'''
device = "cpu"
# if torch.cuda.is_available():
#     device = "cuda:0"
#     torch.cuda.set_device(0)
# logger.info("使用的设备是：%s", device)



def main(config):
    # 加载模型和tokenizer
    dataset = get_dataset()
    model, tokenizer = load(config)
    model, peft_config = load_fine_tuning_model(model, config)
    train(model, tokenizer, dataset, peft_config, config)


def train(model, tokenizer, dataset, peft_config, config):
    training_arguments = SFTConfig(
        dataset_text_field="text",
        packing=False,
        max_seq_length=2048,
        output_dir="./output/phi-2-fine-tuning",
        num_train_epochs=1,
        gradient_accumulation_steps=1,
        optim="sgd",
        save_strategy="epoch",
        logging_steps=100,
        logging_strategy="steps",
        learning_rate=2e-4,
        fp16=False,
        bf16=False,
        group_by_length=True,
        disable_tqdm=False,
        report_to="none",
    )
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        args=training_arguments,
        tokenizer=tokenizer,
    )

    trainer.train()


def eval():
    pass


if __name__ == "__main__":
    main(config)
