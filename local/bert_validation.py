"""Script to score locally a finetuned Bert model on a validation set using (among others) the Jaccard-based F0.5 score which I couldn't manage to include in the training pipeline.
"""
#%%
import sys, os

sys.path.append(os.path.abspath("../usr/lib/"))
for script_folder in os.listdir("../usr/lib/"):
    sys.path.append(os.path.abspath("../usr/lib/" + script_folder))

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

from coleridgeutils import coleridge_fscore, clean_text

# %% Load training args
model_path = "../output/bert-finetuning_kaggle/output/"
training_args = torch.load(model_path + "training_args.bin")
#%% Get the datasets
train_file = "../input/coleridgebiluodownsampled/ner_train_downsampled.json"
validation_file = "../input/coleridgebiluodownsampled/ner_val.json"

data_files = {}
data_files["train"] = train_file
data_files["validation"] = validation_file
extension = train_file.split(".")[-1]
datasets = load_dataset(extension, data_files=data_files)

column_names = datasets["validation"].column_names
features = datasets["validation"].features
text_column_name = "tokens" if "tokens" in column_names else column_names[0]
label_column_name = f"ner_tags" if f"ner_tags" in column_names else column_names[1]

# In the event the labels are not a `Sequence[ClassLabel]`, we will need to go through the dataset to get the
# unique labels.
def get_label_list(labels):
    unique_labels = set()
    for label in labels:
        unique_labels = unique_labels | set(label)
    label_list = list(unique_labels)
    label_list.sort()
    return label_list


# features[label_column_name] will be different for json and csv files:
# json files: Sequence(feature=Value(dtype='string', id=None), length=-1, id=None)
# csv files: Value(dtype='string', id=None)
if hasattr(features[label_column_name], "feature") and isinstance(
    features[label_column_name].feature, ClassLabel
):
    # if isinstance(features[label_column_name].feature, ClassLabel):
    label_list = features[label_column_name].feature.names
    # No need to convert the labels since they are already ints.
    label_to_id = {i: i for i in range(len(label_list))}
else:
    label_list = get_label_list(datasets["train"][label_column_name])
    label_to_id = {l: i for i, l in enumerate(label_list)}
num_labels = len(label_list)

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


tokenized_datasets = datasets.map(
    tokenize_and_align_labels,
    batched=True,
    num_proc=None,
    load_from_cache_file=True,
)

# Data collator
data_collator = DataCollatorForTokenClassification(tokenizer)

# Metrics
def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    # Remove ignored index (special tokens)
    true_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    return {
        "accuracy_score": accuracy_score(true_labels, true_predictions),
        "precision": precision_score(true_labels, true_predictions),
        "recall": recall_score(true_labels, true_predictions),
        "f1": f1_score(true_labels, true_predictions),
    }


#%%
# Initialize our Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)
# %%
test_dataset = tokenized_datasets["validation"]
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
updated_dataset = datasets["validation"].add_column("bert_tags", true_predictions)
updated_dataset = updated_dataset.add_column("bert_probas", true_probas)
#%%
import spacy
from spacy.training import biluo_tags_to_spans

nlp = spacy.load("en_core_web_sm")
nlp.select_pipes(enable="")
#%% save predicted dataset labels to csv
malformed_count = 0
preds = []
truths = []
publications_labels = {}
for example in updated_dataset:
    doc = nlp(example["text"])
    try:
        # use iob_to_biluo before calling biluo_tags_to_spans
        ents_span = biluo_tags_to_spans(doc, example["bert_tags"])
        doc.set_ents(ents_span)
    except:
        malformed_count += 1

    entities = set([clean_text(ent.text) for ent in doc.ents])
    if example["Id"] in publications_labels:
        publications_labels[example["Id"]].update(entities)
    else:
        publications_labels[example["Id"]] = entities.copy()

    preds.append(list(entities))
    ents_span = biluo_tags_to_spans(doc, example["ner_tags"])
    doc.set_ents(ents_span)
    entities = [clean_text(ent.text) for ent in doc.ents]
    truths.append(entities)

predictions_df = []
for key, val in publications_labels.items():
    predictions_df.append({"Id": key, "PredictionString": "|".join(val)})

#%%
coleridge_fscore(preds, truths)
#%%
pd.DataFrame(predictions_df).set_index("Id").to_csv("bert-predictions.csv")
#%% save predicted tags to JSONL
updated_dataset.to_json("bert-predictions.json", orient="records", lines=True)
#%%
output_test_results_file = os.path.join(training_args.output_dir, "test_results.txt")
if trainer.is_world_process_zero():
    with open(output_test_results_file, "w") as writer:
        for key, value in metrics.items():
            logger.info(f"  {key} = {value}")
            writer.write(f"{key} = {value}\n")
