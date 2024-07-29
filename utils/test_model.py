import configparser
import logging.config
import os

import torch
from transformers import pipeline

'''
使用logging
'''
logging.config.fileConfig('../config/logging.ini')
logger = logging.getLogger()

'''
CUDA报OOM错误
'''
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
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
        model="../model/phi-2/",
        device_map="auto",
        trust_remote_code=True,
    )
    prompt = "You are an expert in sentiment classification. You will receive a piece of text and a list of candidate options, and you are asked to select the sentiment that matches the emotional tendency of the text from the candidates."
    user = "text:i was feeling a little skeptical that it would arrive on time the situation was not improved by the fact that despite various perfect party setups seeking in ffxi nobody was bothering to set them up including me but duh im lazy, options:['sadness', 'joy', 'love', 'anger', 'fear', 'surprise']"
    label = "fear"


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
    config.read("../config/config.ini")
    test_env(config)
