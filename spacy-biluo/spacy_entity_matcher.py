#%%
import subprocess, sys, os


def upgrade(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-U", package])


loc = os.environ.get("KAGGLE_KERNEL_RUN_TYPE", "Localhost")
config = {}
if loc == "Batch":
    config["path_to_csv"] = "../input/coleridgeinitiative-show-us-the-data/train.csv"
    config[
        "path_to_publications"
    ] = "../input/coleridgeinitiative-show-us-the-data/train"
    upgrade("spacy")
    subprocess.check_call([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])

else:
    # config["path_to_csv"] = "../input/subset_pub-split/data/train.csv"
    # config["path_to_publications"] = "../input/subset_pub-split/data/train/"
    config["path_to_csv"] = "../input/coleridgeinitiative-show-us-the-data/train.csv"
    config[
        "path_to_publications"
    ] = "../input/coleridgeinitiative-show-us-the-data/train"


#%%
import json
from pathlib import Path

import pandas as pd
import numpy as np
import nltk
import spacy
from spacy.training.iob_utils import doc_to_biluo_tags
from tqdm import tqdm

#%%
iob_map = {0: "", 1: "I-D", 2: "O", 3: "B-D"}


def publication_iob(publication_path, dataset_labels):
    """Tokenize and find occurrences of dataset_labels in the publication texts.
    output: jsonl of the publication with keys 'Id', 'section_title', 'sentence', 'tokens', 'ner_tags' (IOB2)
    """
    nlp = spacy.load("en_core_web_sm")
    nlp.select_pipes(enable="")
    ruler = nlp.add_pipe("entity_ruler")
    patterns = [{"label": "DATASET", "pattern": label} for label in dataset_labels]
    ruler.add_patterns(patterns)

    output = []
    with open(publication_path, "r") as f:
        publication = json.load(f)

    for section in publication:
        entry = {"Id": publication_path.stem, "section_title": section["section_title"]}
        sentences = nltk.tokenize.sent_tokenize(section["text"])
        for sentence in sentences:
            doc = nlp(sentence)
            entry["sentence"] = doc.text
            entry["tokens"] = [token.text for token in doc]
            entry["ner_tags"] = list(map(iob_map.get, doc.to_array("ENT_IOB").tolist()))
            output.append(entry.copy())
    return output


# With "Beginning Postsecondary Student" and "Beginning Postsecondary Students" as matching terms
# and the same two strings in the text, spacy's phraseMatcher reports 2 matching (but we could expect 3),
# but with "Beginning Postsecondary Student" and "Beginning Postsecondary" as matching terms, it reports 4 matching (but we could expect 2 like above).
# the entity ruler seems to deal better with these cases.
# %%


def publication_biluo(publication_path, dataset_labels):
    """Tokenize and find occurrences of dataset_labels in the publication texts.
    output: jsonl of the publication with keys 'Id', 'section_title', 'text', 'tokens', 'ner_tags' (BILUO)
    """
    nlp = spacy.load("en_core_web_sm")
    nlp.max_length = 6_000_000
    # we use the sentencizer to get sentence boundaries
    nlp.select_pipes(enable="")
    nlp.add_pipe("sentencizer")
    ruler = nlp.add_pipe("entity_ruler")
    patterns = [{"label": "DATASET", "pattern": label} for label in dataset_labels]
    ruler.add_patterns(patterns)

    output = []
    with open(publication_path, "r") as f:
        publication = json.load(f)

    for section in publication:
        entry = {"Id": publication_path.stem, "section_title": section["section_title"]}
        doc = nlp(section["text"])
        part_start = 0
        while part_start < len(doc):
            part_end = doc[part_start + 128 : part_start + 129].sent.end
            if part_end - part_start > 255:
                part_end = part_start + 128
                # while last token is part of an entity (we don't want to cut entities)
                while doc[part_end].ent_iob != 2:
                    part_end += 1
            # span will be between 128 and 255 tokens long (except for the last one which can be shorter)
            span = doc[part_start:part_end]
            entry["text"] = span.text
            entry["tokens"] = [token.text for token in span]
            entry["ner_tags"] = doc_to_biluo_tags(span.as_doc())
            entry["ent_count"] = len(span.ents)
            entry["ents"] = [ent.text for ent in span.ents]
            output.append(entry.copy())
            part_start = part_end
    return output


# %%
def process_folder(folder_path, train_df):
    publications = Path(folder_path).iterdir()

    global_df = pd.DataFrame()
    for p in publications:
        labels = np.unique(
            df[df["Id"] == p.stem][["dataset_title", "dataset_label"]].values.ravel()
        )
        output = publication_biluo(p, labels)
        p_df = pd.DataFrame(output)
        global_df = global_df.append(p_df)

        output_folder = Path(folder_path).name
        Path(output_folder).mkdir(exist_ok=True)
        p_df.to_json(
            output_folder + "/" + p.stem + ".jsonl",
            orient="records",
            lines=True,
        )
    global_df.to_json(output_folder + ".json", orient="records", lines=True)


#%%
df = pd.read_csv(config["path_to_csv"])
process_folder(config["path_to_publications"], df)
# %% COMPARISON WITH BERT PRETOKENIZER
# from tokenizers.pre_tokenizers import BertPreTokenizer

# pretokenizer = BertPreTokenizer()
# # pretokenizer.pre_tokenize_str('All told, the annual, in-state cost of attendance at a public four-year institution will run about $38,000 in the early 2030s in today\'s dollars')
# tokenized = pretokenizer.pre_tokenize_str(content[0]["text"])
# print("Spacy:", len(doc), "Hugginface:", len(tokenized))
# # Huggingface splits numbers with a comma whereas Spacy does not.

# %%
