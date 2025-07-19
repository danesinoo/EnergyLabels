import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification


class Tokenizer:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(
            "saattrupdan/nbailab-base-ner-scandi"
        )
        self.model = AutoModelForTokenClassification.from_pretrained(
            "saattrupdan/nbailab-base-ner-scandi"
        )
        self.model.eval()

    def tokenize(self, text):
        # Tokenize text and get necessary mappings
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            return_offsets_mapping=True,
            truncation=True,
            max_length=512,
        )

        input_ids = inputs["input_ids"]
        offset_mapping = inputs["offset_mapping"]
        word_ids = inputs.word_ids(0)  # Word IDs for the first sequence

        # Get last hidden states from the model
        with torch.no_grad():
            outputs = self.model(input_ids, output_hidden_states=True)
            last_hidden_states = outputs.hidden_states[-1][
                0
            ]  # Shape: [seq_len, hidden_dim]

        # Group tokens by word and compute average representations
        words = []
        embeddings = []
        current_word_id = None
        current_embeddings = []
        current_start = None
        current_end = None

        for idx, word_id in enumerate(word_ids):
            if word_id is None:  # Skip special tokens (e.g., [CLS], [SEP])
                continue

            start, end = offset_mapping[0][idx].tolist()

            if word_id == current_word_id:
                # Continue accumulating tokens for the current word
                current_embeddings.append(last_hidden_states[idx])
                current_end = end  # Update word end position
            else:
                # Save previous word if exists
                if current_word_id is not None:
                    word_text = text[current_start:current_end]
                    avg_embed = torch.mean(torch.stack(current_embeddings), dim=0)
                    words.append(word_text)
                    embeddings.append(avg_embed)

                # Start new word
                current_word_id = word_id
                current_embeddings = [last_hidden_states[idx]]
                current_start = start
                current_end = end

        # Add the last word
        if current_word_id is not None:
            word_text = text[current_start:current_end]
            avg_embed = torch.mean(torch.stack(current_embeddings), dim=0)
            words.append(word_text)
            embeddings.append(avg_embed)

        return words, embeddings
