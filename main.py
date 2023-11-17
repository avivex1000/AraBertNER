import os

from src.config import BASE_MODEL_PATH, DATASET_DIR_PATH
from src.model import load_classification_model
from src.trainer import load_base_model, tokenize_dataset, create_trainer

os.environ["TOKENIZERS_PARALLELISM"] = "false"

if __name__ == "__main__":
    labels = ["NoGood", "Good"]
    base_model, tokenizer = load_base_model(BASE_MODEL_PATH, labels)
    dataset = tokenize_dataset(tokenizer, DATASET_DIR_PATH)
    trainer = create_trainer(base_model, dataset, tokenizer)

    # Run the training task
    trainer.train()

    model = load_classification_model()
    print(model("مرحبا، اسمي طال"))
