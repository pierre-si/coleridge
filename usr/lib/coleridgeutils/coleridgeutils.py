import subprocess, sys, os


def install(path, name):
    subprocess.check_call(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "-I",  # ignore (overwrite) the already installed package
            "--no-index",
            "--find-links=" + path,
            name,
        ]
    )


def no_deps_install(path):
    subprocess.check_call(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--no-dependencies",
            path,
        ]
    )


loc = os.environ.get("KAGGLE_KERNEL_RUN_TYPE", "Localhost")
if loc == "Batch" or loc == "Interactive":
    install("../input/coleridge-packages/spacy", "spacy")
    no_deps_install(
        "../input/coleridge-packages/spacy/en_core_web_sm-3.0.0-py3-none-any.whl"
    )

from pathlib import Path
import json
import re

import pandas as pd
import numpy as np
import nltk
import spacy
from spacy.training.iob_utils import doc_to_biluo_tags

#%%
def clean_text(txt):
    return re.sub("[^A-Za-z0-9]+", " ", str(txt).lower()).strip()


# %%
def publications_cleaned_sentences_json(path):
    """Check for all publications in the given folder, split the texts by sentences
    Writes a jsonl file with keys: Id, section_title, sentence, tokens
    """
    publications = path.iterdir()
    rows = []
    for ppath in publications:
        row = {"Id": Path(ppath).stem}

        with open(ppath) as f:
            p = json.load(f)
        for section in p:
            row["section_title"] = section["section_title"]
            sentences = nltk.tokenize.sent_tokenize(section["text"])
            for sentence in sentences:
                sentence = clean_text(sentence)
                row["sentence"] = sentence
                row["tokens"] = sentence.split()
                rows.append(row.copy())
    df = pd.DataFrame(rows)
    df["ner_tags"] = df["tokens"].apply(
        lambda x: ["O"] * len(x)
    )  # add dummy column of ner_tags to match train.json format
    df.to_json("sentences.json", orient="records", lines=True)


def publications_sentences_spacy(path):
    nlp = spacy.load("en_core_web_sm")
    nlp.max_length = 2_000_000
    nlp.select_pipes(enable="")

    path = Path(path)
    publications = path.iterdir()
    rows = []
    for ppath in publications:
        row = {"Id": Path(ppath).stem}

        with open(ppath) as f:
            p = json.load(f)
        for section in p:
            row["section_title"] = section["section_title"]
            sentences = nltk.tokenize.sent_tokenize(section["text"])
            for sentence in sentences:
                row["sentence"] = sentence
                doc = nlp(sentence)
                row["tokens"] = [token.text for token in doc]
                row["ner_tags"] = doc_to_biluo_tags(doc)
                rows.append(row.copy())
    df = pd.DataFrame(rows)
    df.to_json("sentences.json", orient="records", lines=True)


#%%
def predictions_to_array(preds_df):
    """Transform labels predictions to array of int
    preds_df: dataframe of predictions with a string of labels ('O', 'D') in each row
    Output: dataframe of predictions with an array of 0 and 1 in each row
    """
    preds_df["preds"] = preds_df[0].apply(
        lambda x: np.array(x.replace("O", "0").replace("D", "1").split(" ")).astype(
            "bool"
        )
    )
    return preds_df[["preds"]]


#%%
def chunks_indexes(string):
    """Returns the indices of the chunks of 'D' tags in the predictions string.
    This function lets us deal with the case where multiple datasets are cited in a same sentence
    '00DD000DDD00' → [[2, 3],  [7, 8, 9]]
    """
    string = "".join(string.split())
    chunks_idx = []
    chunk_idx = []
    for idx, c in enumerate(string):
        if c == "D":
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
    sentences = pd.read_json(sentences_path, orient="records", lines=True)
    sentences["tokens"] = sentences["tokens"].apply(np.array)

    # retrieve dataset names for each sentence
    preds["chunks"] = preds[0].apply(chunks_indexes)
    df = pd.concat([sentences, preds], axis=1)
    df["PredictionSet"] = df.apply(
        lambda x: set([" ".join(x["tokens"][idx]) for idx in x["chunks"]]), axis=1
    )

    # aggregate by publication
    df = df[["Id", "PredictionSet"]]
    df = df.groupby("Id").agg({"PredictionSet": lambda x: set.union(*x)})
    df["PredictionString"] = df.apply(
        lambda x: "|".join([stri for stri in x["PredictionSet"]]), axis=1
    )
    df = df[["PredictionString"]]
    df.to_csv("submission.csv")


#%%
def jaccard(str1, str2):
    a = set(str1.lower().split())
    b = set(str2.lower().split())
    c = a.intersection(b)
    return float(len(c)) / (len(a) + len(b) - len(c))


#%%
def publications_matches(preds, truths):
    """Computes the number of TP, FP and FN for a publication
    args:
        preds: list of predicted dataset_labels
        truth: list of ground truth dataset_labels
    return:
        true positives, false positives, false negatives
    """
    # predictions have a default jaccard score of 0 (unmatched predictions will keep this value of 0)
    if len(preds) == 0:
        return 0, 0, len(truths)
    preds_jaccard = [0] * len(preds)
    fn = 0
    # looking for the prediction that matches the ground_truth
    for ground_truth in truths:
        match = 0
        match_jaccard = 0
        for i, pred in enumerate(preds):
            jaccard_score = jaccard(ground_truth, pred)
            if jaccard_score > match_jaccard:
                match_jaccard = jaccard_score
                match = i
        # Any ground truths with no nearest predictions are counted as false negatives ('no nearest predictions' is interpreted as: the jaccard score between this ground_truth and the matched prediction is lower than 0.5)
        if match_jaccard < 0.5:
            fn += 1
        # We store the jaccard score for the matched prediction
        # This will be used to count TP and FP
        # A prediction can match multiple ground_truth. We keep the highest jaccard score (predictions matching with a 0.4 and a 0.6 score will thus be counted as TP).
        if match_jaccard > preds_jaccard[match]:
            preds_jaccard[match] = match_jaccard

    # Any matched predictions where the (highest) Jaccard score meets or exceeds the threshold of 0.5 are counted as true positives
    tp = (np.array(preds_jaccard) >= 0.5).sum()
    # The remainder and any unmatched predictions are counted as false positive
    fp = len(preds) - tp
    return tp, fp, fn


def coleridge_fscore(preds, truths):
    """Computes Coleridge's Jaccard-based F0.5 score
    args:
        preds: list of list of predicted dataset_labels
        truths: list of list of ground truth dataset_labels
    """
    tp = 0
    fp = 0
    fn = 0
    for pub_preds, pub_truths in zip(preds, truths):
        ptp, pfp, pfn = publications_matches(pub_preds, pub_truths)
        tp += ptp
        fp += pfp
        fn += pfn

    return (1 + 0.5 ** 2) * tp / ((1 + 0.5 ** 2) * tp + 0.5 ** 2 * fn + fp)


# %%
