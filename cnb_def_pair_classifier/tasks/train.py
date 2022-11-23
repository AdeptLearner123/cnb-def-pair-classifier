from config import TRAIN_DATA, TEST_DATA
from cnb_def_pair_classifier.dataset.dataset import DefPairsDataset

from transformers import AutoTokenizer, DataCollatorWithPadding, AutoModelForSequenceClassification, DataCollatorWithPadding, TrainingArguments, Trainer
import evaluate
import numpy as np


def compute_metrics(eval_preds):
    metric = evaluate.load("f1")
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


def train():
    checkpoint = "bert-large-cased"
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=3)

    train_dataset = DefPairsDataset(TRAIN_DATA)
    test_dataset = DefPairsDataset(TEST_DATA)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    training_args = TrainingArguments("test-trainer", evaluation_strategy="epoch")
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)

    trainer = Trainer(
        model,
        training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    trainer.train()