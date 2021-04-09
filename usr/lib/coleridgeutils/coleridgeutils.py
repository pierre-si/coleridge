#%%
import os
from pathlib import Path
import json
import re

import pandas as pd
import numpy as np
import nltk

#%%
def clean_text(txt):
    return re.sub('[^A-Za-z0-9]+', ' ', str(txt).lower()).strip()
# %%
def publications_cleaned_sentences_json(path):
    """Check for all publications in the given folder, split the texts by sentences
    Writes a jsonl file with keys: Id, section_title, sentence, tokens
    """
    publications = path.iterdir()
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
    df = pd.DataFrame(rows)
    df['ner_tags'] = df['tokens'].apply(lambda x: ["O"]*len(x)) # add dummy column of ner_tags to match train.json format
    df.to_json('sentences.json', orient='records', lines=True)

#%%
def predictions_to_array(preds_df):
    """Transform labels predictions to array of int
    preds_df: dataframe of predictions with a string of labels ('O', 'D') in each row
    Output: dataframe of predictions with an array of 0 and 1 in each row
    """
    preds_df['preds'] = preds_df[0].apply(lambda x: np.array(x.replace('O', '0').replace('D', '1').split(' ')).astype('bool'))
    return preds_df[['preds']]
#%%
def chunks_indexes(string):
    """Returns the indices of the chunks of 'D' tags in the predictions string.
    This function lets us deal with the case where multiple datasets are cited in a same sentence
    '00DD000DDD00' → [[2, 3],  [7, 8, 9]]
    """
    string = ''.join(string.split())
    chunks_idx = []
    chunk_idx = []
    for idx, c in enumerate(string):
        if c == 'D':
            chunk_idx.append(idx)
        # if c is not a dataset tags and we were in a dataset tags chunk i.e. if the chunk has ended
        elif len(chunk_idx) != 0:
            chunks_idx.append(chunk_idx)
            chunk_idx = []
    return chunks_idx
# %%
def predictions_to_submission(sentences_path, predictions_path):
    """Transform predictions from run_ner to submissions for Coleridge
    Input:
    sentences_path: path to jsonl file output by publications_cleaned_sentence_json
    predictions_path: path to the prediction_file output by run_ner --do_predict, trained with 'O' and 'D'
    Write:
    submission.csv: csv with columns "Id" and "PredictionString"
    """
    preds = pd.read_csv(predictions_path, header=None)
    sentences = pd.read_json(sentences_path, orient='records', lines=True)
    sentences['tokens'] = sentences['tokens'].apply(np.array)

    # retrieve dataset names for each sentence
    preds['chunks'] = preds[0].apply(chunks_indexes)
    df = pd.concat([sentences, preds], axis=1)
    df['PredictionSet'] = df.apply(lambda x: set([' '.join(x['tokens'][idx]) for idx in x['chunks']]), axis=1)

    # aggregate by publication
    df = df[['Id', 'PredictionSet']]
    df = df.groupby('Id').agg({'PredictionSet':lambda x: set.union(*x)})
    df['PredictionString'] = df.apply(lambda x: '|'.join([stri for stri in x['PredictionSet']]), axis=1)
    df = df[['PredictionString']]
    df.to_csv('submission.csv')
