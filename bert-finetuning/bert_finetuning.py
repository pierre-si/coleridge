#%%
import os

# Add utility_scripts in the current path so that they can be imported directly just like in interactive mode
sys.path.append(os.path.abspath("../usr/lib/"))
for script_folder in os.listdir("../usr/lib/"):
    sys.path.append(os.path.abspath("../usr/lib/" + script_folder))

from kaggleutils import upgrade

loc = os.environ.get("KAGGLE_KERNEL_RUN_TYPE", "Localhost")
config = {}
if loc == "Batch":
    config["save_total_limit"] = 12
    config[
        "train_file"
    ] = "../input/coleridgebiluodownsampled/ner_train_downsampled.json"
    config["val_file"] = "../input/coleridgebiluodownsampled/ner_val.json"
    config["batch_size"] = 16
    config["num_train_epochs"] = 2
    config["save_steps"] = 100
    upgrade("fsspec")
    upgrade("datasets")
    upgrade("seqeval")

else:
    config["save_total_limit"] = None
    config[
        "train_file"
    ] = "../input/coleridgebiluodownsampled/ner_train_downsampled.json"
    config["val_file"] = "../input/coleridgebiluodownsampled/ner_val_short.json"
    # config["train_file"] = "../input/subset_pub-split/biluo/ner_train_downsampled.json"
    # config["val_file"] = "../input/subset_pub-split/biluo/ner_val.json"
    config["batch_size"] = 4
    config["num_train_epochs"] = 0.4
    config["save_steps"] = 100

#%%
import logging

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

logger = logging.getLogger(__name__)
# %% Load training args
model_path = "distilbert-base-cased"
training_args = TrainingArguments(
    output_dir="output/",
    overwrite_output_dir=False,
    do_train=True,
    do_eval=True,
    do_predict=False,
    evaluation_strategy=IntervalStrategy.STEPS,
    prediction_loss_only=False,
    per_device_train_batch_size=config["batch_size"],
    per_device_eval_batch_size=config["batch_size"],
    gradient_accumulation_steps=1,
    eval_accumulation_steps=None,
    learning_rate=5e-05,
    weight_decay=0.0,
    adam_beta1=0.9,
    adam_beta2=0.999,
    adam_epsilon=1e-08,
    max_grad_norm=1.0,
    num_train_epochs=config["num_train_epochs"],
    max_steps=-1,
    lr_scheduler_type=SchedulerType.LINEAR,
    warmup_ratio=0.0,
    warmup_steps=0,
    # logging_dir="runs/",
    logging_strategy=IntervalStrategy.STEPS,
    logging_first_step=True,
    logging_steps=config["save_steps"],
    save_strategy=IntervalStrategy.STEPS,
    save_steps=config["save_steps"],
    save_total_limit=config["save_total_limit"],
    no_cuda=False,
    seed=42,
    fp16=False,
    fp16_opt_level="O1",
    fp16_backend="auto",
    fp16_full_eval=False,
    local_rank=-1,
    tpu_num_cores=None,
    tpu_metrics_debug=False,
    debug="",
    dataloader_drop_last=False,
    eval_steps=config["save_steps"],
    dataloader_num_workers=0,
    past_index=-1,
    run_name="output/",
    disable_tqdm=False,
    remove_unused_columns=True,
    label_names=None,
    load_best_model_at_end=False,
    metric_for_best_model=None,
    greater_is_better=None,
    ignore_data_skip=False,
    sharded_ddp=[],
    deepspeed=None,
    label_smoothing_factor=0.0,
    adafactor=False,
    group_by_length=False,
    length_column_name="length",
    report_to=["tensorboard"],
    ddp_find_unused_parameters=None,
    dataloader_pin_memory=True,
    skip_memory_metrics=False,
)

#%%
if (
    os.path.exists(training_args.output_dir)
    and os.listdir(training_args.output_dir)
    and training_args.do_train
    and not training_args.overwrite_output_dir
):
    raise ValueError(
        f"Output directory ({training_args.output_dir}) already exists and is not empty."
        "Use --overwrite_output_dir to overcome."
    )

#%% Get the datasets

data_files = {}
if config["train_file"] is not None:
    data_files["train"] = config["train_file"]
if config["val_file"] is not None:
    data_files["validation"] = config["val_file"]

extension = config["train_file"].split(".")[-1]
datasets = load_dataset(extension, data_files=data_files)

if config["val_file"] is None:
    datasets = datasets["train"].train_test_split(test_size=0.1)
    datasets["validation"] = datasets.pop("test")

datasets["train"] = datasets["train"].shuffle(seed=30)

column_names = datasets["train"].column_names
features = datasets["train"].features
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
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)
#%% Training
trainer.train()
trainer.save_model()  # Saves the tokenizer too for easy upload
# %% Evaluation
# results = {}
# logger.info("*** Evaluate ***")

# results = trainer.evaluate()

# output_eval_file = os.path.join(
#     training_args.output_dir, "eval_results_ner.txt"
# )
# if trainer.is_world_process_zero():
#     with open(output_eval_file, "w") as writer:
#         logger.info("***** Eval results *****")
#         for key, value in results.items():
#             logger.info(f"  {key} = {value}")
#             writer.write(f"{key} = {value}\n")
