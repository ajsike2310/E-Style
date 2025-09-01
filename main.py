from pipeline.model import ClothingTransformer
from pipeline.dataset import ClothingDataset
from pipeline.train import train_model
from pipeline.inference import recommend
from transformers import BertTokenizer
import pandas as pd

def load_and_label_data():
    bottoms = pd.read_csv("data/bottoms.csv").dropna(subset=["details"])
    bottoms["category"] = 0  # 0 for bottoms
    tops = pd.read_csv("data/tops.csv").dropna(subset=["details"])
    tops["category"] = 1    # 1 for tops
    data = pd.concat([bottoms, tops], ignore_index=True)
    data = data[["details", "category"]].rename(columns={"category": "labels"})
    return data.reset_index(drop=True)

if __name__ == "__main__":
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    data = load_and_label_data()
    dataset = ClothingDataset(data, tokenizer)
    model = ClothingTransformer(num_classes=2)
    train_model(model, dataset)
    # Try a bottom and a top description
    print("Prediction for jeans:", recommend(model, tokenizer, "Straight fit jeans with faded effect"))
    print("Prediction for blazer:", recommend(model, tokenizer, "Straight-fit blazer with a notched lapel collar and long sleeves with buttoned cuffs."))
