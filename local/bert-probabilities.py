#%%
"""Notebook to see how to retrieve the probabilities associated to a Bert for token classification model"""
#%%
import numpy as np
from datasets import ClassLabel, load_dataset
from seqeval.metrics import accuracy_score, f1_score, precision_score, recall_score
from transformers import BertForTokenClassification, BertTokenizerFast, DataCollatorForTokenClassification, Trainer
#%%
datasets = load_dataset('json', data_files='../input/smaller/iob2/ner_val_short.json')
text_column_name = 'tokens'
label_column_name = 'ner_tags'
# %%
tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
model = BertForTokenClassification.from_pretrained('../bert-finetuning/output/')
data_collator = DataCollatorForTokenClassification(tokenizer)
# %%
def get_label_list(labels):
    unique_labels = set()
    for label in labels:
        unique_labels = unique_labels | set(label)
    label_list = list(unique_labels)
    label_list.sort()
    return label_list

label_list = get_label_list(datasets["train"][label_column_name])
label_to_id = {l: i for i, l in enumerate(label_list)}
num_labels = len(label_list)

# Tokenize all texts and align the labels with them.
def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(
        examples[text_column_name],
        padding=False,#padding,
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
                label_ids.append(-100)#label_to_id[label[word_idx]] if data_args.label_all_tokens else -100)
            previous_word_idx = word_idx

        labels.append(label_ids)
    tokenized_inputs["labels"] = labels
    return tokenized_inputs

tokenized_datasets = datasets.map(
    tokenize_and_align_labels,
    batched=True,
#    num_proc=data_args.preprocessing_num_workers,
#    load_from_cache_file=not data_args.overwrite_cache,
)

# %%
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
# %%
trainer = Trainer(
    model=model,
    #args=training_args,
    #train_dataset=tokenized_datasets["train"] if training_args.do_train else None,
    #eval_dataset=tokenized_datasets["validation"] if training_args.do_eval else None,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# %%
test_dataset = tokenized_datasets["train"]
predictions, labels, metrics = trainer.predict(test_dataset)

# %%
import torch
from torch.nn.functional import softmax
# %%
softmax(torch.tensor(predictions), dim=2)