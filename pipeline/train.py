import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm import tqdm

def train_model(model, dataset, epochs=3, lr=5e-5, batch_size=16):
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = AdamW(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()

    model.train()
    for epoch in range(epochs):
        loop = tqdm(dataloader, leave=True)
        for batch in loop:
            optimizer.zero_grad()
            outputs = model(batch['input_ids'], batch['attention_mask'])
            labels = batch['labels'] if 'labels' in batch else torch.zeros(outputs.size(0), dtype=torch.long)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            loop.set_description(f"Epoch {epoch+1}")
            loop.set_postfix(loss=loss.item())
