import torch
from torch.utils.data import Dataset

class ClothingDataset(Dataset):
    def __init__(self, df, tokenizer, max_len=128):
        self.df = df
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        text = self.df.iloc[idx]["details"] if "details" in self.df.columns else self.df.iloc[idx]["text"]
        inputs = self.tokenizer(text, padding="max_length", truncation=True, max_length=self.max_len, return_tensors="pt")
        item = {key: val.squeeze(0) for key, val in inputs.items()}
        if "labels" in self.df.columns:
            item["labels"] = torch.tensor(self.df.iloc[idx]["labels"], dtype=torch.long)
        return item
