from swanlab.integration.huggingface import SwanLabCallback


def load_monitor(config):
    return SwanLabCallback(
        project="",
        description="微调",
        config={
        }
    )
