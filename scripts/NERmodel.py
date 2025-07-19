import torch.nn as nn
import torch.nn.functional as F


class TokenClassifier(nn.Module):
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
        x = F.relu(self.fc2(x))
        flat_logits = self.output(x)
        logits = flat_logits.view(batch_size, seq_len, self.num_classes)
        return logits
