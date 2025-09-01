import pandas as pd
from transformers import BertTokenizer
from pipeline.pair_dataset import TopBottomPairDataset
from pipeline.pair_model import PairClassifier
from pipeline.pair_train import train_pair_model
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

if __name__ == "__main__":
    from sklearn.model_selection import train_test_split
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    data = pd.read_csv("data/top_bottom_pairs.csv")
    # Shuffle and stratify split
    train_df, val_df = train_test_split(data, test_size=0.3, random_state=42, stratify=data['label'])
    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    train_dataset = TopBottomPairDataset(train_df, tokenizer)
    val_dataset = TopBottomPairDataset(val_df, tokenizer)
    model = PairClassifier(num_classes=2)
    train_pair_model(model, train_dataset, epochs=10, batch_size=4)

    # Evaluate on validation set
    y_true, y_pred = [], []
    print("\nPredictions for each pair in validation set:")
    for i in range(len(val_dataset)):
        item = val_dataset[i]
        label = item['labels'].item()
        inputs = {k: v.unsqueeze(0) for k, v in item.items() if k in ['input_ids', 'attention_mask']}
        pred = model(**inputs).argmax(dim=1).item()
        y_true.append(label)
        y_pred.append(pred)
        print(f"Pair {i+1}: Top='{val_df.iloc[i]['top_product_name']}', Bottom='{val_df.iloc[i]['bottom_product_name']}' | True: {label} | Pred: {pred}")
    print("\nValidation Evaluation:")
    print("Accuracy:", accuracy_score(y_true, y_pred))
    print("Confusion matrix:\n", confusion_matrix(y_true, y_pred))
    print("Classification report:\n", classification_report(y_true, y_pred))
