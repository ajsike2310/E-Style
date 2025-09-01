from pipeline.model import ClothingTransformer
from pipeline.dataset import ClothingDataset
from pipeline.train import train_model
from pipeline.inference import recommend
from transformers import BertTokenizer
import pandas as pd

def load_and_label_data(balance=True):
    bottoms = pd.read_csv("data/bottoms.csv").dropna(subset=["details", "product_name"])
    bottoms["category"] = 0  # 0 for bottoms
    tops = pd.read_csv("data/tops.csv").dropna(subset=["details", "product_name"])
    tops["category"] = 1    # 1 for tops
    if balance:
        min_len = min(len(bottoms), len(tops))
        bottoms = bottoms.sample(min_len, random_state=42)
        tops = tops.sample(min_len, random_state=42)
    data = pd.concat([bottoms, tops], ignore_index=True)
    # Combine product_name and details for richer input
    data["text"] = data["product_name"].astype(str) + ". " + data["details"].astype(str)
    data = data[["text", "category"]].rename(columns={"category": "labels"})
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

    print("\nSample test set descriptions and labels:")
    print(test_data.head(10)[["text", "labels"]])
    print("\nTrain set class distribution:")
    print(train_data['labels'].value_counts())
    print("Test set class distribution:")
    print(test_data['labels'].value_counts())
    print("First 20 test set labels:")
    print(test_data['labels'].head(20).tolist())

    train_dataset = ClothingDataset(train_data.rename(columns={"text": "details"}), tokenizer)
    test_dataset = ClothingDataset(test_data.rename(columns={"text": "details"}), tokenizer)
    model = ClothingTransformer(num_classes=2)

    # Train for more epochs
    train_model(model, train_dataset, epochs=10)

    # Evaluate on test set and print predictions vs true labels
    y_true, y_pred = [], []
    print("\nSample predictions vs true labels:")
    for i in range(min(20, len(test_dataset))):
        item = test_dataset[i]
        label = item['labels'].item() if 'labels' in item else 0
        pred = recommend(model, tokenizer, test_data.iloc[i]['text'])
        y_true.append(label)
        y_pred.append(pred)
        print(f"Text: {test_data.iloc[i]['text'][:80]}... | True: {label} | Pred: {pred}")
    # Evaluate on all test set
    for i in range(20, len(test_dataset)):
        item = test_dataset[i]
        label = item['labels'].item() if 'labels' in item else 0
        pred = recommend(model, tokenizer, test_data.iloc[i]['text'])
        y_true.append(label)
        y_pred.append(pred)
    print("\nEvaluation on test set:")
    print("Accuracy:", accuracy_score(y_true, y_pred))
    print("Confusion matrix:\n", confusion_matrix(y_true, y_pred))
    print("Classification report:\n", classification_report(y_true, y_pred))

    # Try a bottom and a top description
    print("Prediction for jeans:", recommend(model, tokenizer, "STRAIGHT FIT JEANS. Straight-leg jeans with a five-pocket design. Faded with ripped details and a stain print on the legs. Front zip fly and top button fastening."))
    print("Prediction for blazer:", recommend(model, tokenizer, "BASIC BLAZER. Straight-fit blazer with a notched lapel collar and long sleeves with buttoned cuffs. Patch pockets at the hip and an inside pocket detail. Back vents at the hem. Buttoned front."))
