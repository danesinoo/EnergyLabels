import re
import numpy as np


def extract_tags(tagged_sentence, tokenized_words):
    """
    Extract tags for each word in the tokenized list based on the tagged sentence.

    Args:
        tagged_sentence: String with tagged words in format {word}_{tag}
        tokenized_words: List of tokenized words

    Returns:
        numpy array of logit arrays, one for each word in tokenized_words
    """

    # Define label mapping
    labels = [
        "None",
        "Pieces",
        "Manufacturer",
        "SubType",
        "HxType",
        "NominalEffectEach",
        "Year",
    ]
    label_to_index = {label: i for i, label in enumerate(labels)}

    # Initialize result array with 'None' tags for all words
    num_words = len(tokenized_words)
    num_labels = len(labels)
    result = np.zeros((num_words, num_labels), dtype=int)
    result[:, 0] = 1  # Set all to 'None' initially

    # Extract tagged words and their labels from the tagged sentence
    tagged_pattern = r"\{([^}]+)\}_\{([^}]+)\}"
    tagged_matches = re.findall(tagged_pattern, tagged_sentence)

    # Create a mapping from word to tag
    word_to_tag = {}
    for word, tag in tagged_matches:
        word_to_tag[word] = tag

    # Update tags for words that are tagged
    for i, word in enumerate(tokenized_words):
        if word in word_to_tag:
            tag = word_to_tag[word]
            if tag in label_to_index:
                # Reset the row and set the appropriate tag
                result[i, :] = 0
                result[i, label_to_index[tag]] = 1

    return result


def tag_sentence(tokenized_words, word_tags):
    """
    Create a tagged sentence from tokenized words and their tags.

    Args:
        original_sentence: The original sentence (for reference)
        tokenized_words: List of tokenized words
        word_tags: numpy array of one-hot encoded tags for each word

    Returns:
        String with tagged words in format {word}_{tag}
    """

    # Define label mapping
    labels = [
        "None",
        "Pieces",
        "Manufacturer",
        "SubType",
        "HxType",
        "NominalEffectEach",
        "Year",
    ]

    tagged_words = []

    for i, word in enumerate(tokenized_words):
        # Get the tag index (position of 1 in the one-hot array)
        tag_index = np.argmax(word_tags[i])
        tag = labels[tag_index]

        if tag == "None":
            # No tag, just use the word
            tagged_words.append(word)
        else:
            # Tag exists, format as {word}_{tag}
            tagged_words.append(f"{{{word}}}_{{{tag}}}")

    # Join words with spaces
    return " ".join(tagged_words)
