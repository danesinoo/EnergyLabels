# Import Dataset
import pandas as pd
import pandas as pd
from tqdm.notebook import trange

d_gt = pd.read_excel("../data/data_district_heating.xlsx", sheet_name="Ground Truth")
d_gt["text"] = (
    d_gt["S_text"].fillna("").astype(str) + ". " + d_gt["L_text"].fillna("").astype(str)
)
d_gt["SingleHx"] = d_gt.iloc[:, 13:31].isna().all(axis=1)
d_gt.head()
# Create Dataset
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


class SequenceDataset(Dataset):
    def __init__(self, tokens_list, attention_mask, labels_list):
        """
        Args:
            tokens_list: List of token embeddings (each item is a tensor of shape [seq_len, embedding_dim])
            attention_mask: List of attention masks (each item is a tensor of shape [seq_len])
            labels_list: List of labels (each item is a tensor of shape [len(tokens_list), 1])
        """
        self.tokens = tokens_list
        self.attention_mask = attention_mask
        self.labels = labels_list

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, idx):
        return self.tokens[idx], self.attention_mask[idx], self.labels[idx]


model_name = "Qwen/Qwen3-0.6B"
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
tokenizer = AutoTokenizer.from_pretrained(model_name)
X = tokenizer(
    d_gt["text"].tolist(),
    return_tensors="pt",
    max_length=512,
    truncation=True,
    padding=True,
)
y = torch.tensor(d_gt["SingleHx"].astype(int).tolist())
X_train, X_test, X_attention_mask, y_attention_mask, y_train, y_test = train_test_split(
    X["input_ids"], X["attention_mask"], y, test_size=0.2, random_state=42
)

train_data = SequenceDataset(X_train, X_attention_mask, y_train)
eval_data = SequenceDataset(X_test, X_attention_mask, y_test)
model.config.pad_token_id = tokenizer.pad_token_id
model.config.pad_token = tokenizer.pad_token
from tqdm import tqdm


def train(model, X, y, epochs=3):
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    train_loader = DataLoader(
        SequenceDataset(X["input_ids"], X["attention_mask"], y),
        batch_size=16,
        shuffle=True,
    )
    losses = []

    for epoch in range(epochs):
        loss = 0
        optimizer.zero_grad()
        data_iter = iter(train_loader)
        pbar = tqdm(range(len(data_iter)), desc="Training Epochs")
        for i in pbar:
            input_ids, attention_mask, labels = next(data_iter)
            outputs = model(
                input_ids=input_ids, attention_mask=attention_mask, labels=labels
            )
            loss += outputs.loss

            if i % 5 == 0:
                loss = loss / 5
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                pbar.set_postfix({"loss": loss.item()})
                losses.append(loss.item())
                loss = 0

    return losses


losses = train(
    model, {"input_ids": X_train, "attention_mask": X_attention_mask}, y_train, epochs=3
)
