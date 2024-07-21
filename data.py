from datasets import load_dataset




def get_dataset():
    # dataset = load_dataset(path="data/emotion_classification")
    dataset = load_dataset("hieunguyenminh/roleplay", split="train[0:1000]")
    return dataset



