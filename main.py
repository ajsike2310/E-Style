from pipeline.model import ClothingTransformer
from pipeline.dataset import ClothingDataset
from pipeline.train import train_model
from pipeline.inference import recommend
from transformers import BertTokenizer
import pandas as pd

if __name__ == "__main__":
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    data = pd.read_csv("data/sample.csv")  # Add your own data

    dataset = ClothingDataset(data, tokenizer)
    model = ClothingTransformer(num_classes=2)

    train_model(model, dataset)
    print(recommend(model, tokenizer, "Red floral dress for wedding"))
