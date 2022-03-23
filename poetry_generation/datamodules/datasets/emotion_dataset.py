import pandas as pd
from torch.utils.data import Dataset

label2int = {"bad": 1, "good": 0, "neutral": 2}


class EmotionDataset(Dataset):
    def __init__(self, data_path) -> None:
        super().__init__()
        self.data_column = "prompt_pl"
        self.class_column = "sentiment_idx"
        df = pd.read_csv(data_path, delimiter="\t")

        self.df = df[[self.data_column, self.class_column]]

    def __getitem__(self, idx) -> tuple(str, int):
        return (self.df.loc[idx, self.data_column], self.df.loc[idx, self.class_column])

    def __len__(self) -> int:
        return self.df.shape[0]
