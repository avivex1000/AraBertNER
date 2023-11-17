from typing import List

import evaluate

import numpy as np


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    accuracy = evaluate.load("accuracy")
    return accuracy.compute(predictions=predictions, references=labels)


def id2label_format(labels: List[str]):
    return {i: label for i, label in enumerate(labels)}


def label2id_format(labels: List[str]):
    return {label: i for i, label in enumerate(labels)}
