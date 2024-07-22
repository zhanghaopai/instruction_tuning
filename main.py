import configparser
import os

import swanlab
from trl import SFTTrainer, SFTConfig

from data import get_dataset
from load import load_model, load_lora_model, load_tokenizer
from monitor.monitor import load_monitor
from utils.logger import get_logger

config = configparser.ConfigParser()
config.read("config/config.ini")
logger = get_logger(config)

'''
设置环境
'''
os.environ["BNB_CUDA_VERSION"] = "121"


def main(config):
    # 加载模型和tokenizer
    dataset = get_dataset()
    tokenizer = load_tokenizer(config)
    model = load_model(config)
    model = load_lora_model(model, config)
    callback_func = load_monitor(config)
    train(model, tokenizer, dataset, callback_func)


def train(model, tokenizer, dataset, callback_func):
    training_arguments = SFTConfig(
        dataset_text_field="text",
        packing=False,
        max_seq_length=1024,
        output_dir="./output/phi-2-fine-tuning",
        num_train_epochs=1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=16,
        optim="paged_adamw_8bit",  # 分页实现内存管理，否则可能出现oom
        save_strategy="epoch",
        logging_steps=1,
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
        callbacks=[callback_func]
    )
    trainer.train()
    swanlab.finish()


def eval():
    pass


if __name__ == "__main__":
    main(config)
