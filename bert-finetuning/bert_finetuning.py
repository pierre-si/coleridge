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
    # HfArgumentParser,
    # PreTrainedTokenizerFast,
    Trainer,
    TrainingArguments,
    # set_seed,
)
from transformers.training_args import IntervalStrategy, SchedulerType

logger = logging.getLogger(__name__)
# %% Load training args
model_path = "bert-base-uncased"
training_args = TrainingArguments(
    output_dir="output/",
    overwrite_output_dir=False,
    do_train=True,
    do_eval=True,
    do_predict=False,
    evaluation_strategy=IntervalStrategy.STEPS,
    prediction_loss_only=False,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=1,
    eval_accumulation_steps=None,
    learning_rate=5e-05,
    weight_decay=0.0,
    adam_beta1=0.9,
    adam_beta2=0.999,
    adam_epsilon=1e-08,
    max_grad_norm=1.0,
    num_train_epochs=3.0,
    max_steps=-1,
    lr_scheduler_type=SchedulerType.LINEAR,
    warmup_ratio=0.0,
    warmup_steps=0,
    logging_dir="runs/Apr30_16-42-31_debian",
    logging_strategy=IntervalStrategy.STEPS,
    logging_first_step=False,
    logging_steps=500,
    save_strategy=IntervalStrategy.STEPS,
    save_steps=500,
    save_total_limit=None,
    no_cuda=False,
    seed=42,
    fp16=False,
    fp16_opt_level="O1",
    fp16_backend="auto",
    fp16_full_eval=False,
    local_rank=-1,
    tpu_num_cores=None,
    tpu_metrics_debug=False,
    debug=False,
    dataloader_drop_last=False,
    eval_steps=500,
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
)  # , _n_gpu=1)#, mp_parameters=)

#%% Get the datasets
train_file = "../input/subset/biluo/ner_train.json"  # 5,1% de ner_train (lui-même 2,1% des publications)
validation_file = "../input/subset/biluo/ner_val.json"  # 14,5% de ner_val (lui même 0,7% des publications)

data_files = {}
if train_file is not None:
    data_files["train"] = train_file
if validation_file is not None:
    data_files["validation"] = validation_file

extension = train_file.split(".")[-1]
datasets = load_dataset(extension, data_files=data_files)

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
trainer.train(model_path=model_path if os.path.isdir(model_path) else None)
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

# %% Predict
logger.info("*** Predict ***")

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
    doc = nlp(example["sentence"])
    try:
        # use iob_to_biluo before calling biluo_tags_to_spans
        ents_span = biluo_tags_to_spans(doc, example["bert_tags"])
        doc.set_ents(ents_span)
    except:
        pass

    entities = set([ent.text for ent in doc.ents])
    if example["Id"] in publications_labels:
        publications_labels[example["Id"]].update(entities)
    else:
        publications_labels[example["Id"]] = entities.copy()

predictions_df = []
for key, val in publications_labels.items():
    predictions_df.append({"Id": key, "PredictionString": "|".join(val)})

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

#%%
# Save predictions
output_test_predictions_file = os.path.join(
    training_args.output_dir, "test_predictions.txt"
)
if trainer.is_world_process_zero():
    with open(output_test_predictions_file, "w") as writer:
        for prediction in true_predictions:
            writer.write(" ".join(prediction) + "\n")
# %%
# DataTrainingArguments(task_name='ner', dataset_name=None, dataset_config_name=None, train_file='../input/coleridge-sentences/ner_train-256.json', validation_file='../input/coleridge-sentences/ner_val-256.json', test_file='sentences.json', overwrite_cache=False, preprocessing_num_workers=None, pad_to_max_length=False, label_all_tokens=False)

# num_labels: 2

# ModelArguments(model_name_or_path='../input/bert-finetuned', config_name=None, tokenizer_name=None, cache_dir=None)
