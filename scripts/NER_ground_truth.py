import pandas as pd
import numpy as np
import data_from_xlsx
from tqdm import trange
from tokenizer import Tokenizer


def contains(word, words):
    return word in words


def ner_data(file_name="data/data_district_heating.xlsx"):
    tokenizer = Tokenizer()

    labels = [
        "None",
        "Pieces",
        "Manufacturer",
        "SubType",
        "HxType",
        "NominalEffectEach",
        "Year",
    ]

    symbols = [".", "i", "er", "veksler", "et", "varme", "var", "af"]
    labels_to_num = np.eye(7)
    d_gt = pd.read_excel(file_name, sheet_name="Ground Truth")
    # %%
    pieces = (
        d_gt.iloc[:, [7, 13, 19, 25]]
        .fillna("")
        .astype(str)
        .agg(" ".join, axis=1)
        .str.lower()
    )
    manufacturers = (
        d_gt.iloc[:, [8, 14, 20, 26]]
        .fillna("")
        .astype(str)
        .agg(" ".join, axis=1)
        .str.lower()
    )
    subtype = (
        d_gt.iloc[:, [9, 15, 21, 27]]
        .fillna("")
        .astype(str)
        .agg(" ".join, axis=1)
        .str.lower()
    )
    hxType = (
        d_gt.iloc[:, [10, 16, 22, 28]]
        .fillna("")
        .astype(str)
        .agg(" ".join, axis=1)
        .str.lower()
    )
    nominalEffectEach = (
        d_gt.iloc[:, [11, 17, 23, 29]]
        .fillna("")
        .astype(str)
        .agg(" ".join, axis=1)
        .str.lower()
    )
    year = (
        d_gt.iloc[:, [12, 18, 24, 30]]
        .fillna("")
        .astype(str)
        .agg(" ".join, axis=1)
        .str.lower()
    )
    # %%
    ground_truth = pd.DataFrame()
    ground_truth["text"] = (
        d_gt[["S_text", "L_text"]].fillna("").astype(str).agg(" ".join, axis=1)
    )
    # %%
    ground_truth["words"] = pd.NA
    ground_truth["tokens"] = pd.NA
    ground_truth["labels"] = pd.NA

    for i in trange(len(ground_truth)):
        words, tokens, _ = tokenizer.tokenize(ground_truth.loc[i, "text"])
        words = [word.lower() for word in words]
        ground_truth.at[i, "words"] = words
        ground_truth.at[i, "tokens"] = tokens

        labels = labels_to_num[0] * np.ones((len(words), 7))

        for j, word in enumerate(words):
            if contains(word, symbols):
                continue
            elif contains(word, pieces[i]):
                labels[j] = labels_to_num[1]
            elif contains(word, manufacturers[i]):
                labels[j] = labels_to_num[2]
            elif contains(word, subtype[i]):
                labels[j] = labels_to_num[3]
            elif contains(word, hxType[i]):
                labels[j] = labels_to_num[4]
            elif contains(word, nominalEffectEach[i]):
                labels[j] = labels_to_num[5]
            elif contains(word, year[i]):
                labels[j] = labels_to_num[6]

        ground_truth.at[i, "labels"] = labels

    return ground_truth


def nested_ner_data(file_name="data/data_district_heating.xlsx"):
    tokenizer = Tokenizer()

    labels = [
        "None",
        "Pieces",
        "Manufacturer",
        "SubType",
        "HxType",
        "NominalEffectEach",
        "Year",
    ]

    symbols = [".", "i", "er", "veksler", "et", "varme", "var", "af"]
    labels_to_num = np.eye(7)
    d_gt = pd.read_excel(file_name, sheet_name="Ground Truth")
    # %%
    pieces = (
        d_gt.iloc[:, [7, 13, 19, 25]]
        .fillna("")
        .astype(str)
        .agg(" ".join, axis=1)
        .str.lower()
    )
    manufacturers = (
        d_gt.iloc[:, [8, 14, 20, 26]]
        .fillna("")
        .astype(str)
        .agg(" ".join, axis=1)
        .str.lower()
    )
    subtype = (
        d_gt.iloc[:, [9, 15, 21, 27]]
        .fillna("")
        .astype(str)
        .agg(" ".join, axis=1)
        .str.lower()
    )
    hxType = (
        d_gt.iloc[:, [10, 16, 22, 28]]
        .fillna("")
        .astype(str)
        .agg(" ".join, axis=1)
        .str.lower()
    )
    nominalEffectEach = (
        d_gt.iloc[:, [11, 17, 23, 29]]
        .fillna("")
        .astype(str)
        .agg(" ".join, axis=1)
        .str.lower()
    )
    year = (
        d_gt.iloc[:, [12, 18, 24, 30]]
        .fillna("")
        .astype(str)
        .agg(" ".join, axis=1)
        .str.lower()
    )
    # %%
    ground_truth = pd.DataFrame()
    ground_truth["text"] = (
        d_gt[["S_text", "L_text"]].fillna("").astype(str).agg(" ".join, axis=1)
    )
    # %%
    ground_truth["words"] = pd.NA
    ground_truth["tokens"] = pd.NA
    ground_truth["labels"] = pd.NA

    for i in trange(len(ground_truth)):
        words, tokens, _ = tokenizer.tokenize(ground_truth.loc[i, "text"])
        words = [word.lower() for word in words]
        ground_truth.at[i, "words"] = words
        ground_truth.at[i, "tokens"] = tokens

        labels = labels_to_num[0] * np.ones((len(words), 7))

        for j, word in enumerate(words):
            if contains(word, symbols):
                continue
            elif contains(word, pieces[i]):
                labels[j] = labels_to_num[1]
            elif contains(word, manufacturers[i]):
                labels[j] = labels_to_num[2]
            elif contains(word, subtype[i]):
                labels[j] = labels_to_num[3]
            elif contains(word, hxType[i]):
                labels[j] = labels_to_num[4]
            elif contains(word, nominalEffectEach[i]):
                labels[j] = labels_to_num[5]
            elif contains(word, year[i]):
                labels[j] = labels_to_num[6]

        ground_truth.at[i, "labels"] = labels

    return ground_truth
