import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import trange
from torch.nn.utils.rnn import pad_sequence
from sklearn.base import BaseEstimator, ClassifierMixin
import pandas as pd
import numpy as np


def collate_fn(batch):
    """
    Custom collate function to handle variable-length sequences using torch's pad_sequence
    """
    tokens_list, labels_list = zip(*batch)

    # Pad tokens and labels using torch's built-in function
    padded_tokens = pad_sequence(tokens_list, batch_first=True, padding_value=0.0)
    padded_labels = pad_sequence(labels_list, batch_first=True, padding_value=0.0)

    # Create attention masks
    attention_masks = torch.zeros(len(tokens_list), padded_tokens.shape[1])
    for i, tokens in enumerate(tokens_list):
        attention_masks[i, : len(tokens)] = 1

    return padded_tokens, padded_labels, attention_masks


class TokenDataset(Dataset):
    def __init__(self, tokens_list, labels_list):
        """
        Args:
            tokens_list: List of token embeddings (each item is a tensor of shape [seq_len, embedding_dim])
            labels_list: List of labels (each item is a tensor of shape [seq_len, 7])
        """
        self.tokens_list = tokens_list
        self.labels_list = labels_list

    def __len__(self):
        return len(self.tokens_list)

    def __getitem__(self, idx):
        tokens = torch.stack(self.tokens_list[idx])
        labels = torch.from_numpy(self.labels_list[idx])
        return tokens, labels


class NeuralNetwork(nn.Module):
    def __init__(self, embedding_dim, hidden_dim=49, num_classes=7):
        """
        Args:
            embedding_dim: Dimension of input token embeddings
            hidden_dim: Size of hidden layer (0 for no hidden layer)
            num_classes: Number of output classes (default=7)
        """
        super().__init__()
        self.num_classes = num_classes
        self.fc1 = nn.Linear(embedding_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, num_classes)

    def forward(self, token_embeddings):
        """
        Args:
            token_embeddings: Tensor of shape (batch_size, seq_len, embedding_dim)
        Returns:
            logits: Tensor of shape (batch_size, seq_len, num_classes)
        """
        batch_size, seq_len, emb_dim = token_embeddings.shape
        flat_embeddings = token_embeddings.view(-1, emb_dim)
        x = F.relu(self.fc1(flat_embeddings))
        x = F.sigmoid(self.fc2(x))
        flat_logits = self.output(x)
        logits = flat_logits.view(batch_size, seq_len, self.num_classes)
        return logits


class TokenClassifier(BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        embedding_dim=768,
        num_classes=7,
        nn=NeuralNetwork,
        initial_batch_size=16,
        max_batch_size=64,
        num_epochs=1000,
        eval_split=0.2,
        seed=42,
        lr=0.001,
    ):
        # Store all parameters as attributes (required for sklearn)
        self.embedding_dim = embedding_dim
        self.num_classes = num_classes
        self.nn = nn
        self.initial_batch_size = initial_batch_size
        self.max_batch_size = max_batch_size
        self.num_epochs = num_epochs
        self.eval_split = eval_split
        self.seed = seed
        self.lr = lr

        # Internal attributes
        self.model = None
        self.is_fitted_ = False
        self.training_results_ = None

    def _initialize_model(self):
        """Initialize the neural network model"""
        if self.nn is None:
            raise ValueError("Neural network class (nn) must be provided")
        self.model = self.nn(self.embedding_dim, num_classes=self.num_classes)

    def fit(self, X, y):
        """
        Fit the token classifier.

        Args:
            X: list/array of tokens
            y: labels
        Returns:
            self: Returns the instance itself
        """
        self._initialize_model()
        train_dataset = TokenDataset(X, y)

        criterion = nn.BCEWithLogitsLoss(reduction="none")
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        train_losses = []
        train_accs = []
        current_batch_size = self.initial_batch_size

        print(f"Starting training with batch size {current_batch_size}")
        self.is_fitted_ = True
        pbar = trange(self.num_epochs)
        for epoch in pbar:
            train_loader = DataLoader(
                train_dataset,
                batch_size=current_batch_size,
                shuffle=True,
                collate_fn=collate_fn,
            )

            self.model.train()
            epoch_loss = 0.0
            num_batches = 0

            for tokens, labels, attention_mask in train_loader:
                optimizer.zero_grad()

                logits = self.model(tokens)

                loss_per_token = criterion(logits, labels)
                loss_per_token = loss_per_token * attention_mask.unsqueeze(-1)
                loss = loss_per_token.sum() / attention_mask.sum()

                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                num_batches += 1

            avg_epoch_loss = epoch_loss / num_batches

            pbar.set_postfix(
                {
                    "Train Loss": f"{avg_epoch_loss:.4f}",
                    "Batch Size": f"{current_batch_size}",
                }
            )

            if avg_epoch_loss < 0.01 and current_batch_size < self.max_batch_size:
                current_batch_size = min(current_batch_size * 2, self.max_batch_size)

        self.training_results_ = {
            "train_losses": train_losses,
            "train_accuracies": train_accs,
            "model": self.model,
        }

        return self

    def predict(self, X, batch_size=32):
        """
        Predict class labels for samples in X.

        Args:
            X: list/array of tokens (same format as fit method)
            batch_size: batch size for prediction (default: 32)

        Returns:
            list: Predicted labels for each token sequence
        """
        if not self.is_fitted_:
            raise ValueError(
                "This classifier has not been fitted yet. Call 'fit' first."
            )

        dummy_labels = [np.zeros_like(token_list) for token_list in X]
        predict_dataset = TokenDataset(X, dummy_labels)

        # Create data loader using the same collate_fn as in fit
        predict_loader = DataLoader(
            predict_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_fn,
        )

        predictions = []
        self.model.eval()

        with torch.no_grad():
            for tokens, labels, attention_mask in predict_loader:
                logits = self.model(tokens)
                pred = torch.argmax(logits, dim=-1)

                for i in range(pred.size(0)):
                    seq_pred = pred[i]
                    seq_mask = attention_mask[i]
                    valid_predictions = seq_pred[seq_mask.bool()].cpu().numpy().tolist()
                    predictions.append(valid_predictions)

        return predictions

    def score(self, X, y=None):
        """
        Return the mean accuracy on the given test data and labels.

        Args:
            X: Test samples
            y: True labels

        Returns:
            float: Mean accuracy score
        """
        if not self.is_fitted_:
            raise ValueError(
                "This classifier has not been fitted yet. Call 'fit' first."
            )

        # Create evaluation dataset and loader
        eval_dataset = TokenDataset(X, y)
        eval_loader = DataLoader(
            eval_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn
        )

        return self.eval(eval_loader)

    def eval(self, eval_loader):
        """Evaluate the model on the given data loader"""
        if not self.is_fitted_:
            raise ValueError(
                "This classifier has not been fitted yet. Call 'fit' first."
            )

        acc_sum = 0.0
        total_tokens = 0

        self.model.eval()
        with torch.no_grad():
            for tokens, labels, attention_mask in eval_loader:
                logits = self.model(tokens)

                pred = torch.argmax(logits, dim=-1)
                target = torch.argmax(labels, dim=-1)

                acc_per_token = (pred == target).float() * attention_mask
                acc_sum += acc_per_token.sum().item()
                total_tokens += attention_mask.sum().item()

        acc = acc_sum / total_tokens if total_tokens > 0 else 0.0
        return acc

    def get_params(self, deep=True):
        """
        Get parameters for this estimator.

        Args:
            deep (bool): If True, return parameters for sub-estimators too.

        Returns:
            dict: Parameter names mapped to their values.
        """
        return {
            "embedding_dim": self.embedding_dim,
            "num_classes": self.num_classes,
            "nn": self.nn,
            "initial_batch_size": self.initial_batch_size,
            "max_batch_size": self.max_batch_size,
            "num_epochs": self.num_epochs,
            "eval_split": self.eval_split,
            "seed": self.seed,
            "lr": self.lr,
        }

    def set_params(self, **params):
        """
        Set the parameters of this estimator.

        Args:
            **params: Estimator parameters.

        Returns:
            self: Estimator instance.
        """
        valid_params = set(self.get_params().keys())
        for key, value in params.items():
            if key not in valid_params:
                raise ValueError(
                    f"Invalid parameter {key} for estimator {type(self).__name__}"
                )
            setattr(self, key, value)

        # Reset fitted state when parameters change
        self.is_fitted_ = False
        self.model = None
        self.training_results_ = None

        return self

    def save_model(self, file_name):
        """
        Save the trained model to a file.

        Args:
            file_name (str): Path where to save the model
        """
        if not self.is_fitted_:
            raise ValueError("Cannot save model that hasn't been fitted yet.")

        try:
            # Save the model state dict along with architecture parameters
            save_dict = {
                "state_dict": self.model.state_dict(),
                "embedding_dim": self.embedding_dim,
                "num_classes": self.num_classes,
                "nn_class": self.nn.__name__ if self.nn else None,
            }
            torch.save(save_dict, file_name)
            print(f"Model saved successfully to {file_name}")
        except Exception as e:
            print(f"Error saving model: {e}")
            raise

    def load_model(self, file_name, nn_class=None):
        """
        Load a previously saved model.

        Args:
            file_name (str): Path to the saved model
            nn_class: Neural network class to instantiate (optional if saved with model)
        """
        try:
            save_dict = torch.load(file_name)

            # Use saved architecture parameters or class parameters
            embedding_dim = save_dict.get("embedding_dim", self.embedding_dim)
            num_classes = save_dict.get("num_classes", self.num_classes)

            if nn_class is None:
                nn_class = self.nn
            if nn_class is None:
                raise ValueError(
                    "Neural network class must be provided either in constructor or as parameter"
                )

            # Initialize model architecture
            self.model = nn_class(embedding_dim, num_classes=num_classes)
            # Load the saved state dict
            self.model.load_state_dict(save_dict["state_dict"])
            self.model.eval()  # Set to evaluation mode
            self.is_fitted_ = True
            print(f"Model loaded successfully from {file_name}")
        except Exception as e:
            print(f"Error loading model: {e}")
            raise


if __name__ == "__main__":
    from NER_ground_truth import ner_data
    import plot

    ner_gt = ner_data("data/data_district_heating.xlsx")
    model = TokenClassifier()
    model.fit(ner_gt["tokens"].values, ner_gt["labels"].values)
    dict_res = model.training_results_
    # model.save_model("models/NER_token_classifier")
    losses = {
        "Train loss": [1 - acc for acc in dict_res["train_accuracies"]],
    }

    plot.axis(losses, "Loss on training of `TokenClassifier`", "Epochs", "Loss")
