from transformers import pipeline

from src.config import OUTPUT_MODEL_PATH


def load_classification_model():
    return pipeline(
        "text-classification",  # Don't change this as it is specific for the classification task!
        model=OUTPUT_MODEL_PATH,
    )

