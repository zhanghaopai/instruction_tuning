import configparser
import os

import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split

id2emotion_dict = {"0": "sadness", "1": "joy", "2": "love", "3": "anger", "4": "fear", "5": "surprise"}
emotion2id_dict = {"sadness": "0", "joy": "1", "love": "2", "anger": "3", "fear": "4", "surprise": "5"}


def get_dataset(config):
    origin_ds_path = os.path.join(
        config.root_path, "assets", "sa", "origin"
    )
    instruction_ds_path = os.path.join(
        config.root_path, "assets", "sa", "instruction"
    )
    # 如果没有做指令转换，则做指令转换
    if len(os.listdir(instruction_ds_path)) == 0:
        converter(origin_ds_path, instruction_ds_path)
    # 加载情感分析任务的指令数据集
    return load_dataset(path=instruction_ds_path)


# 将原始的情感分类数据集转换为大模型微调可以使用的数据集
def converter(origin_path, instruction_path):
    converted = []
    files = os.listdir(origin_path)
    for file in files:
        sa_pd = pd.read_csv(os.path.join(origin_path, file))
        # 对于每一行数据，进行转换
        for row in range(len(sa_pd)):
            text = sa_pd.iloc[row, :]["text"]
            label = sa_pd.iloc[row, :]["label"]
            message = {
                "instruction": "You are an expert in sentiment classification. " +
                               "You will receive a piece of text and a list of candidate options, " +
                               "and you are asked to select the sentiment " +
                               "that matches the emotional tendency of the text from the candidates.",
                "input": f"text:{text}, options:{emotion2id_dict.keys()}",
                "output": id2emotion_dict[str(label)]
            }
            converted.append(message)
    train_set, test_set = train_test_split(converted, test_size=0.2, random_state=7)
    pd.DataFrame(train_set).to_csv(os.path.join(instruction_path, "train.csv"), mode='w')
    pd.DataFrame(test_set).to_csv(os.path.join(instruction_path, "test.csv"), mode='w')


if __name__ == '__main__':
    config = configparser.ConfigParser()
    config.root_path = "../"
    get_dataset(config)
