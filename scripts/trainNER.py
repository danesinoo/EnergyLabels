import pandas as pd
import numpy as np
import json
from typing import Dict, List, Tuple, Optional
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from collections import defaultdict
import random


class NERTrainingPipeline:
    def __init__(
        self,
        tokenizer,
        token_classifier: nn.Module,
        chatbot,
        data_source: pd.DataFrame,
        label_to_idx: Dict[str, int],
        batch_size: int = 32,
    ):
        """
        Initialize the training pipeline.

        Args:
            tokenizer: Tokenizer that returns (words, tokens)
            token_classifier: Neural network for token classification
            chatbot: LLM chatbot for verification
            data_source: DataFrame with text samples
            label_to_idx: Mapping from label names to indices
        """
        self.tokenizer = tokenizer
        self.token_classifier = token_classifier
        self.chatbot = chatbot
        self.data_source = data_source
        self.label_to_idx = label_to_idx
        self.idx_to_label = {v: k for k, v in label_to_idx.items()}
        self.batch_size = batch_size

        # Cache for tokenized sentences
        self.tokenization_cache = {}

        # Store verified predictions
        self.verified_data = pd.DataFrame(
            columns=["text", "words", "tokens", "labels", "label_indices"]
        )

        # Setup optimizer
        self.optimizer = torch.optim.Adam(token_classifier.parameters(), lr=1e-3)
        self.criterion = nn.CrossEntropyLoss(ignore_index=-1)

    def tokenize_with_cache(self, text: str) -> Tuple[List[str], torch.Tensor]:
        """Tokenize text and cache the result."""
        if text not in self.tokenization_cache:
            words, tokens = self.tokenizer(text)
            self.tokenization_cache[text] = (words, tokens)
        return self.tokenization_cache[text]

    def create_prompt(self, text: str) -> str:
        """Create prompt for LLM verification."""
        prompt = f"""Given the following text, extract and tag each word with one of these labels:
- None (not relevant)
- Pieces (quantity)
- Manufacturer (company name)
- SubType (model identifier)
- HxType (heat exchanger type)
- NominalEffectEach (power/effect)
- Year (year or time period)

Text: "{text}"

Please tag each word in the format: word/LABEL

Example: "1/Pieces stk./None Reci/Manufacturer"

Tagged text:"""
        return prompt

    def parse_llm_tags(self, tagged_text: str, words: List[str]) -> List[str]:
        """Parse LLM output to extract tags."""
        tags = ["None"] * len(words)

        # Split the tagged text
        tagged_parts = tagged_text.strip().split()

        word_idx = 0
        for part in tagged_parts:
            if "/" in part:
                word, tag = part.rsplit("/", 1)
                # Find matching word in original words list
                for i in range(word_idx, len(words)):
                    if words[i].lower() == word.lower():
                        if tag in self.label_to_idx:
                            tags[i] = tag
                        word_idx = i + 1
                        break

        return tags

    def get_nn_predictions(self, tokens: torch.Tensor) -> List[str]:
        """Get neural network predictions for tokens."""
        self.token_classifier.eval()
        with torch.no_grad():
            logits = self.token_classifier(tokens.unsqueeze(0))  # Add batch dimension
            predictions = torch.argmax(logits, dim=-1).squeeze(0)

        return [self.idx_to_label[idx.item()] for idx in predictions]

    def verify_predictions(
        self, text: str, words: List[str], nn_predictions: List[str]
    ) -> Tuple[bool, List[str]]:
        """Verify NN predictions with LLM."""
        # Create tagged sentence for LLM
        tagged_for_llm = tag_sentence(words, nn_predictions)

        # Query LLM
        prompt = self.create_prompt(text)
        llm_response = self.chatbot.query(prompt)

        # Parse LLM tags
        llm_tags = self.parse_llm_tags(llm_response, words)

        # Check agreement
        agreement = all(
            nn_pred == llm_tag for nn_pred, llm_tag in zip(nn_predictions, llm_tags)
        )

        return agreement, llm_tags

    def process_batch(self, batch_texts: List[str]) -> Dict:
        """Process a batch of texts."""
        results = {"verified": [], "disagreements": []}

        for text in batch_texts:
            # Tokenize with cache
            words, tokens = self.tokenize_with_cache(text)

            # Get NN predictions
            nn_predictions = self.get_nn_predictions(tokens)

            # Verify with LLM
            agreement, llm_tags = self.verify_predictions(text, words, nn_predictions)

            if agreement:
                # Convert labels to indices for training
                label_indices = [self.label_to_idx[label] for label in nn_predictions]

                # Store verified data
                new_row = pd.DataFrame(
                    [
                        {
                            "text": text,
                            "words": words,
                            "tokens": tokens.cpu().numpy(),
                            "labels": nn_predictions,
                            "label_indices": label_indices,
                        }
                    ]
                )
                self.verified_data = pd.concat(
                    [self.verified_data, new_row], ignore_index=True
                )

                results["verified"].append(
                    {"text": text, "predictions": nn_predictions}
                )
            else:
                results["disagreements"].append(
                    {
                        "text": text,
                        "nn_predictions": nn_predictions,
                        "llm_predictions": llm_tags,
                    }
                )

        return results

    def train_on_verified_data(self, epochs: int = 5):
        """Train the neural network on verified data."""
        if len(self.verified_data) == 0:
            print("No verified data available for training")
            return

        # Create dataset
        dataset = VerifiedDataset(self.verified_data)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        self.token_classifier.train()

        for epoch in range(epochs):
            total_loss = 0
            for batch in dataloader:
                tokens = batch["tokens"]
                labels = batch["labels"]

                # Forward pass
                logits = self.token_classifier(tokens)

                # Reshape for loss calculation
                logits = logits.view(-1, len(self.label_to_idx))
                labels = labels.view(-1)

                # Calculate loss
                loss = self.criterion(logits, labels)

                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()

            avg_loss = total_loss / len(dataloader)
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")

    def run_training_loop(
        self, num_iterations: int = 10, samples_per_iteration: int = 10
    ):
        """Run the main training loop."""
        for iteration in range(num_iterations):
            print(f"\nIteration {iteration + 1}/{num_iterations}")

            # Extract random samples
            sample_indices = random.sample(
                range(len(self.data_source)),
                min(samples_per_iteration, len(self.data_source)),
            )
            batch_texts = self.data_source.iloc[sample_indices]["text"].tolist()

            # Process batch
            results = self.process_batch(batch_texts)

            print(f"Verified: {len(results['verified'])}")
            print(f"Disagreements: {len(results['disagreements'])}")

            # Train on verified data if we have enough
            if len(self.verified_data) >= self.batch_size:
                print("Training on verified data...")
                self.train_on_verified_data(epochs=3)

            # Save checkpoint
            if (iteration + 1) % 5 == 0:
                self.save_checkpoint(f"checkpoint_{iteration + 1}.pt")

    def save_checkpoint(self, filename: str):
        """Save model checkpoint and verified data."""
        torch.save(
            {
                "model_state_dict": self.token_classifier.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "verified_data": self.verified_data.to_dict(),
                "tokenization_cache": self.tokenization_cache,
            },
            filename,
        )
        print(f"Checkpoint saved to {filename}")


class VerifiedDataset(Dataset):
    """Dataset for verified NER data."""

    def __init__(self, dataframe: pd.DataFrame):
        self.data = dataframe

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        return {
            "tokens": torch.tensor(row["tokens"], dtype=torch.float32),
            "labels": torch.tensor(row["label_indices"], dtype=torch.long),
        }


# Usage example
def main():
    # Initialize components (assuming they're already implemented)
    tokenizer = YourTokenizer()  # Your tokenizer implementation

    # Define label mapping
    label_to_idx = {
        "None": 0,
        "Pieces": 1,
        "Manufacturer": 2,
        "SubType": 3,
        "HxType": 4,
        "NominalEffectEach": 5,
        "Year": 6,
    }

    # Initialize token classifier
    token_classifier = TokenClassifier(
        input_dim=768,  # Adjust based on your token embedding size
        hidden_dim=256,
        num_classes=len(label_to_idx),
    )

    # Initialize chatbot with prompt function
    chatbot = YourChatbot(layout_prompt_func=lambda x: x)

    # Load your data
    data_df = pd.read_csv("your_data.csv")  # Assuming you have a CSV with 'text' column

    # Create and run pipeline
    pipeline = NERTrainingPipeline(
        tokenizer=tokenizer,
        token_classifier=token_classifier,
        chatbot=chatbot,
        data_source=data_df,
        label_to_idx=label_to_idx,
        batch_size=32,
    )

    # Run training loop
    pipeline.run_training_loop(num_iterations=20, samples_per_iteration=10)

    # Access verified data
    print(f"\nTotal verified samples: {len(pipeline.verified_data)}")
    pipeline.verified_data.to_csv("verified_ner_data.csv", index=False)


if __name__ == "__main__":
    main()
