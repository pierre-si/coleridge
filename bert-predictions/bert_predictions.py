import subprocess, sys, os


def local_install(path, name):
    subprocess.check_call(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "-I",
            "--no-index",
            "--find-links=" + path,
            name,
        ]
    )


loc = os.environ.get("KAGGLE_KERNEL_RUN_TYPE", "Localhost")
if loc == "Batch" or loc == "Interactive":
    local_install("../input/coleridge-packages/datasets", "datasets")
    local_install("../input/coleridge-packages/seqeval", "seqeval")

#%%
# Add utility_scripts in the current path so that they can be imported directly just like in interactive mode
sys.path.append(os.path.abspath("../usr/lib/"))
for script_folder in os.listdir("../usr/lib/"):
    sys.path.append(os.path.abspath("../usr/lib/" + script_folder))

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from datasets import ClassLabel, load_dataset
from seqeval.metrics import accuracy_score, f1_score, precision_score, recall_score
from scipy.special import softmax
import transformers
from transformers import (
    AutoConfig,
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments,
)
from transformers.training_args import IntervalStrategy, SchedulerType

from coleridgeutils import (
    publications_sentences_spacy,
    predictions_to_submission,
    clean_text,
)

# %% Load training args
model_path = "../input/bertfinetuned/output/"
training_args = torch.load(model_path + "training_args.bin")
# %% Convert publications test folder to jsonl sentences file
publications_sentences_spacy("../input/coleridgeinitiative-show-us-the-data/test")
# Get the datasets
data_files = {}
data_files["test"] = "sentences.json"
extension = "json"
datasets = load_dataset(extension, data_files=data_files)

column_names = datasets["test"].column_names
features = datasets["test"].features
text_column_name = "tokens" if "tokens" in column_names else column_names[0]
label_column_name = f"ner_tags" if f"ner_tags" in column_names else column_names[1]

num_labels = 5
label_list = ["B-DATASET", "I-DATASET", "L-DATASET", "O", "U-DATASET"]
label_to_id = {l: i for i, l in enumerate(label_list)}

#%% Load pretrained model and tokenizer
config = AutoConfig.from_pretrained(
    model_path,
    num_labels=num_labels,
    finetuning_task="ner",
    cache_dir=None,
)
tokenizer = AutoTokenizer.from_pretrained(
    model_path,
    cache_dir=None,
    use_fast=True,
)
model = AutoModelForTokenClassification.from_pretrained(
    model_path,
    from_tf=None,
    config=config,
    cache_dir=None,
)

# %%
# Preprocessing the dataset
# Padding strategy
padding = ["max_length", False][1]
label_all_tokens = [True, False][1]

# Tokenize all texts and align the labels with them.
def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(
        examples[text_column_name],
        padding=padding,
        truncation=True,
        # We use this argument because the texts in our dataset are lists of words (with a label for each word).
        is_split_into_words=True,
    )
    labels = []
    for i, label in enumerate(examples[label_column_name]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            # Special tokens have a word id that is None. We set the label to -100 so they are automatically
            # ignored in the loss function.
            if word_idx is None:
                label_ids.append(-100)
            # We set the label for the first token of each word.
            elif word_idx != previous_word_idx:
                label_ids.append(label_to_id[label[word_idx]])
            # For the other tokens in a word, we set the label to either the current label or -100, depending on
            # the label_all_tokens flag.
            else:
                label_ids.append(
                    label_to_id[label[word_idx]] if label_all_tokens else -100
                )
            previous_word_idx = word_idx

        labels.append(label_ids)
    tokenized_inputs["labels"] = labels
    return tokenized_inputs


# ABORTED TENTATIVE TO AVOID HAVING TO RELY ON THE LABEL COLUMN
# def tokenize_and_align_labels(examples):
#     tokenized_inputs = tokenizer(
#         examples[text_column_name],
#         padding=padding,
#         truncation=True,
#         # We use this argument because the texts in our dataset are lists of words (with a label for each word).
#         is_split_into_words=True,
#     )
#     labels = []
#     print(len(examples))
#     print('\n')
#     for i in range(len(examples)):
#         word_ids = tokenized_inputs.word_ids(batch_index=i)
#         previous_word_idx = None
#         label_ids = []
#         for word_idx in word_ids:
#             # Special tokens have a word id that is None. We set the label to -100 so they are automatically
#             # ignored in the loss function.
#             if word_idx is None:
#                 label_ids.append(0)
#             # We set the label for the first token of each word.
#             elif word_idx != previous_word_idx:
#                 label_ids.append(1)
#             # For the other tokens in a word, we set the label to either the current label or -100, depending on
#             # the label_all_tokens flag.
#             else:
#                 label_ids.append(
#                     1 if label_all_tokens else 0
#                 )
#             previous_word_idx = word_idx
#         print(len(label_ids))
#         labels.append(label_ids)
#     tokenized_inputs["labels"] = labels
#     return tokenized_inputs


tokenized_datasets = datasets.map(
    tokenize_and_align_labels,
    batched=True,
    num_proc=None,
    load_from_cache_file=True,
)

# Data collator
data_collator = DataCollatorForTokenClassification(tokenizer)

#%%
# Initialize our Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    tokenizer=tokenizer,
    data_collator=data_collator,
)
# %% Predict
test_dataset = tokenized_datasets["test"]
predictions, labels, metrics = trainer.predict(test_dataset)
#%%
probas = softmax(predictions, axis=2)
predictions = np.argmax(predictions, axis=2)

# Remove ignored index (special tokens)
true_predictions = [
    [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
    for prediction, label in zip(predictions, labels)
]

true_probas = [
    [prob[p] for (p, prob, l) in zip(prediction, proba, label) if l != -100]
    for prediction, proba, label in zip(predictions, probas, labels)
]

# %% add predicted tags and associated probabilities to the dataset
# updated_dataset = datasets['test'].map(lambda example: {'bert_tags': })
updated_dataset = datasets["test"].add_column("bert_tags", true_predictions)
updated_dataset = updated_dataset.add_column("bert_probas", true_probas)
#%%
import spacy
from spacy.training import biluo_tags_to_spans

nlp = spacy.load("en_core_web_sm")
nlp.select_pipes(enable="")
#%% save predicted dataset labels to csv
publications_labels = {}
for example in updated_dataset:
    doc = nlp(example["text"])
    try:
        ents_span = biluo_tags_to_spans(doc, example["bert_tags"])
        doc.set_ents(ents_span)
    except:
        pass

    entities = set([clean_text(ent.text) for ent in doc.ents])
    if example["Id"] in publications_labels:
        publications_labels[example["Id"]].update(entities)
    else:
        publications_labels[example["Id"]] = entities.copy()

predictions_df = []
for key, val in publications_labels.items():
    predictions_df.append({"Id": key, "PredictionString": "|".join(val)})

pd.DataFrame(predictions_df).set_index("Id").to_csv("submission.csv")
