import pandas as pd

def merge_columns(s_text, l_text, sep="<sep>"):
    null_idx = s_text.isnull()
    s_text.iloc[null_idx] = ""
    null_idx = l_text.isnull()
    l_text.iloc[null_idx] = ""
    return s_text + sep + l_text

def to_csv(data, file_name, sep = ","):
    data.to_csv(
        "../data/" + file_name + ".csv",
        index=False,
        encoding='utf-8',
        sep=sep,
        header = True
    )

def tag(text, labels):
    tagged_str = text
    tag_columns = [col for col in labels.index]
    for tag_col in tag_columns:
        if pd.notna(text):
            tag_val = labels[tag_col]
            tagged_str = tagged_str.replace(tag_val, f"{tag_val}_{{{tag_col}}}")
    return tagged_str