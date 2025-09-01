from pipeline.model import ClothingTransformer
from pipeline.dataset import ClothingDataset
from pipeline.train import train_model
from pipeline.inference import recommend
from transformers import BertTokenizer
import pandas as pd

def load_and_label_data(balance=True):
    bottoms = pd.read_csv("data/bottoms.csv").dropna(subset=["details"])
    bottoms["category"] = 0  # 0 for bottoms
    tops = pd.read_csv("data/tops.csv").dropna(subset=["details"])
    tops["category"] = 1    # 1 for tops
    if balance:
        min_len = min(len(bottoms), len(tops))
        bottoms = bottoms.sample(min_len, random_state=42)
        tops = tops.sample(min_len, random_state=42)
    data = pd.concat([bottoms, tops], ignore_index=True)
    data = data[["details", "category"]].rename(columns={"category": "labels"})
    return data.reset_index(drop=True)

if __name__ == "__main__":
    from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    data = load_and_label_data(balance=True)
    print("Class distribution after balancing:")
    print(data['labels'].value_counts())

    # Shuffle and split into train/test
    data = data.sample(frac=1, random_state=42).reset_index(drop=True)
    split = int(0.8 * len(data))
    train_data = data.iloc[:split]
    test_data = data.iloc[split:]

    train_dataset = ClothingDataset(train_data, tokenizer)
    test_dataset = ClothingDataset(test_data, tokenizer)
    model = ClothingTransformer(num_classes=2)

    # Train for more epochs
    train_model(model, train_dataset, epochs=6)

    # Evaluate on test set
    y_true, y_pred = [], []
    for i in range(len(test_dataset)):
        item = test_dataset[i]
        label = item['labels'].item() if 'labels' in item else 0
        pred = recommend(model, tokenizer, test_data.iloc[i]['details'])
        y_true.append(label)
        y_pred.append(pred)
    print("\nEvaluation on test set:")
    print("Accuracy:", accuracy_score(y_true, y_pred))
    print("Confusion matrix:\n", confusion_matrix(y_true, y_pred))
    print("Classification report:\n", classification_report(y_true, y_pred))

    # Try a bottom and a top description
    print("Prediction for jeans:", recommend(model, tokenizer, "Straight fit jeans with faded effect"))
    print("Prediction for blazer:", recommend(model, tokenizer, "Straight-fit blazer with a notched lapel collar and long sleeves with buttoned cuffs."))
