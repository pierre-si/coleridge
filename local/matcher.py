#%%
import json
from pathlib import Path

import pandas as pd
import numpy as np
import nltk
import spacy
from spacy.matcher import PhraseMatcher
from tqdm import tqdm
#%%
def publication_iob(publication_path, dataset_labels):
    """Tokenize and find occurrences of dataset_labels in the publication texts.
    output: jsonl of the publication with keys 'section_title', 'tokens', 'ner_tags' (IOB2)
    """
    nlp = spacy.load("en_core_web_sm")
    nlp.select_pipes(enable='')
    ruler = nlp.add_pipe("entity_ruler")
    patterns = [{"label": "DATASET", "pattern": label} for label in dataset_labels]
    ruler.add_patterns(patterns)

    output = []
    with open(publication_path, 'r') as f:
        publication = json.load(f)

    for section in publication:
        entry = {'section_title': section['section_title']}
        sentences = nltk.tokenize.sent_tokenize(section['text'])
        for sentence in sentences:
            doc = nlp(sentence)
            entry['sentence'] = doc.text
            entry['tokens'] = [token.text for token in doc]
            entry['ner_tags'] = doc.to_array('ENT_IOB')
            output.append(entry.copy())
    return output
# With "Beginning Postsecondary Student" and "Beginning Postsecondary Students" as matching terms
# and the same two strings in the text, spacy reports 2 matching (but we could expect 3), 
# but with "Beginning Postsecondary Student" and "Beginning Postsecondary" as matching terms, spacy reports 4 matching (but we could expect 2 like above).
# %%
df = pd.read_csv('../input/smaller/val.csv')
publications = Path('../input/smaller/val/').iterdir()
# %%
for p in tqdm(publications):
    labels = df[df['Id'] == p.stem]['dataset_label'].values
    output = publication_matches(p, labels)
    pd.DataFrame(output).to_json('../input/smaller/iob2/val/'+p.stem+'.jsonl', orient='records', lines=True)
# %% COMPARISON WITH BERT PRETOKENIZER
from tokenizers.pre_tokenizers import BertPreTokenizer  
pretokenizer = BertPreTokenizer()
tokenized = pretokenizer.pre_tokenize_str(content[0]['text'])
print("Spacy:", len(doc), "Hugginface:", len(tokenized))
# Huggingface splits numbers with a comma whereas Spacy does not.
