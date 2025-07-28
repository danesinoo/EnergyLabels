import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification


class Tokenizer:
    def __init__(self, model_name="saattrupdan/nbailab-base-ner-scandi"):
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name
        )
        self.model = AutoModelForTokenClassification.from_pretrained(
            model_name
        )
        self.model.eval()

    def tokenize(self, text):
        """
        Tokenize text and return words, their embeddings, and indices.

        Args:
            text (str): Input text to tokenize

        Returns:
            tuple: (words, embeddings, indices) where:
                - words: List of word strings
                - embeddings: List of averaged token embeddings for each word
                - indices: List of (start, end) character indices for each word
        """
        # Get tokenized inputs and model outputs
        inputs = self._prepare_inputs(text)
        last_hidden_states = self._get_hidden_states(inputs["input_ids"])

        # Extract word information
        word_groups = self._group_tokens_by_word(
            inputs["offset_mapping"][0], inputs.word_ids(0), last_hidden_states
        )

        # Process word groups into final outputs
        words, embeddings = self._process_word_groups(word_groups, text)

        return words, embeddings

    def _prepare_inputs(self, text):
        """Tokenize text and prepare model inputs."""
        return self.tokenizer(
            text,
            return_tensors="pt",
            return_offsets_mapping=True,
            truncation=True,
            max_length=512,
        )

    def _get_hidden_states(self, input_ids):
        """Get last hidden states from the model."""
        with torch.no_grad():
            outputs = self.model(input_ids, output_hidden_states=True)
            return outputs.hidden_states[-1][0]  # Shape: [seq_len, hidden_dim]

    def _group_tokens_by_word(self, offset_mapping, word_ids, hidden_states):
        """Group tokens by their word IDs."""
        word_groups = {}

        for idx, word_id in enumerate(word_ids):
            if word_id is None:  # Skip special tokens
                continue

            start, end = offset_mapping[idx].tolist()

            if word_id not in word_groups:
                word_groups[word_id] = {"embeddings": [], "start": start, "end": end}

            word_groups[word_id]["embeddings"].append(hidden_states[idx])
            word_groups[word_id]["end"] = end  # Update end position

        return word_groups

    def _process_word_groups(self, word_groups, text):
        """Process grouped tokens into words, and embeddings."""
        words = []
        embeddings = []

        # Sort by word_id to maintain order
        for word_id in sorted(word_groups.keys()):
            group = word_groups[word_id]

            # Extract word text
            start_idx = group["start"]
            end_idx = group["end"]
            word_text = text[start_idx:end_idx]

            # Compute average embedding
            avg_embedding = self._average_embeddings(group["embeddings"])

            # Append to results
            words.append(word_text)
            embeddings.append(avg_embedding)

        return words, embeddings

    def _average_embeddings(self, embeddings_list):
        """Compute average of a list of embedding tensors."""
        return torch.mean(torch.stack(embeddings_list), dim=0)