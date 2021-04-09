#%%
import os
from pathlib import Path
import json
import re

import nltk
import pandas as pd
#%%
def clean_text(txt):
    return re.sub('[^A-Za-z0-9]+', ' ', str(txt).lower()).strip()
# %%
def publications_cleaned_sentences_json(path):
    """Check for all publications in the given folder, split the texts by sentences
    Writes a jsonl file with keys: Id, section_title, sentence, tokens
    """
    publications = data_path.iterdir()
    rows = []
    for ppath in publications:
        row = {'Id': Path(ppath).stem}

        with open(ppath) as f:
            p = json.load(f)
        for section in p:
            row['section_title'] = section['section_title']
            sentences = nltk.tokenize.sent_tokenize(section['text'])
            for sentence in sentences:
                sentence = clean_text(sentence)
                row['sentence'] = sentence
                row['tokens'] = sentence.split()
                rows.append(row.copy())
    pd.DataFrame(rows).to_json('sentences.json', orient='records', lines=True)
