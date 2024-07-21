import configparser
import logging.config
import os

import torch
from transformers import pipeline

'''
使用logging
'''
logging.config.fileConfig('config/logging.ini')
logger = logging.getLogger()

'''
CUDA报OOM错误
'''
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
'''
使用CUDA
'''
device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
logger.info("使用的设备是：%s", device)


def test_env(config):
    pipe = pipeline(
        "text-generation",
        model=config.get("base", "offline_model"),
        device_map="auto",
        trust_remote_code=True,
    )
    prompt = "Please write a python program that add numbers from 1 to 10."

    outputs = pipe(
        prompt,
        max_new_tokens=300,
        do_sample=True,
        temperature=0.7,
        top_k=3,
        top_p=0.95,
    )
    logger.debug(outputs[0]["generated_text"])


if __name__ == "__main__":
    config = configparser.ConfigParser()
    config.read("config/config.ini")
    test_env(config)
