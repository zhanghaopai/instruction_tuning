from dataloader.sa_dataloader import get_sa_dataset
import matplotlib.pyplot as plt

emotion2count_dict = {"sadness": 0, "joy": 0, "love": 0, "anger": 0, "fear": 0, "surprise": 0}



if __name__ == '__main__':
    root_path = "../"
    dataset = get_sa_dataset(root_path)
    labels = dataset["test"]["output"]
    for label in labels:
        emotion2count_dict[label]+=1

    x = list(emotion2count_dict.keys())
    y = list(emotion2count_dict.values())

    plt.title("emotion statistics")
    plt.grid(ls="--", alpha=0.5)
    plt.bar(x, y)
    plt.show()
