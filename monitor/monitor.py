from swanlab.integration.huggingface import SwanLabCallback


def load_monitor():
    return SwanLabCallback(
        project="instruction-fine-tuning",
        description="使用twitter情感分类数据微调microsoft/phi-2模型",
    )
