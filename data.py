from datasets import load_dataset


emotions = load_dataset(path="data/emotion_classification")
print(emotions["train"][0])



