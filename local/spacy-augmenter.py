# %%
"""This script aims to augment the training data by matching publication texts with dataset_labels using spacy's phraseMatcher and replacing it with different dataset_labels."""
# %%
from pathlib import Path
import json

from tqdm import tqdm
import pandas as pd
import nltk
import spacy
from spacy.matcher import PhraseMatcher

# %%
def augment_labels(publication_path, dataset_labels, new_dataset_labels):
    """Find occurrences of dataset_labels per sentence in the publication's text fields and replace them with new_dataset_labels.
    output: jsonl of the publication with keys 'section_title', 'text' and 'dataset_label'
    """
    nlp = spacy.load("en_core_web_sm")
    nlp.select_pipes(enable="")
    matcher = PhraseMatcher(nlp.vocab)
    # Only run nlp.make_doc to speed things up
    patterns = [nlp.make_doc(text) for text in dataset_labels]
    matcher.add("TerminologyList", patterns)

    output = []
    with open(publication_path, "r") as f:
        publication = json.load(f)

    for section in publication:
        entry = {"section_title": section["section_title"]}
        sentences = nltk.tokenize.sent_tokenize(section["text"])
        for sentence in sentences:
            doc = nlp(sentence)
            matches = matcher(doc)
            if len(matches) == 0:
                continue
            else:
                for new_label in new_dataset_labels:
                    entry["text"] = (
                        sentence[: doc[matches[0][1]].idx]
                        + new_label
                        + sentence[doc[matches[0][2]].idx :]
                    )
                    entry["dataset_label"] = new_label
                    output.append(entry.copy())
    return output


# %%
df = pd.read_csv("../input/smaller/train.csv")
all_dataset_labels = df["dataset_label"].unique()
publications = Path("../input/smaller/train/").iterdir()
# %%
for p in tqdm(publications):
    labels = df[df["Id"] == p.stem]["dataset_label"].values
    output = augment_labels(p, labels, all_dataset_labels)
    p_df = pd.DataFrame(output)
    p_df.to_json(
        "../input/smaller/augment/" + p.stem + ".jsonl", orient="records", lines=True
    )
# %%
