import platform
print(platform.node())
import socket
print(socket.gethostname())
import os
os.environ["WANDB_DISABLED"] = "true"

from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer

dataset = load_dataset("yelp_review_full", cache_dir="/home/users/uat/data/")
tokenizer = AutoTokenizer.from_pretrained("distilroberta-base", cache_dir="/home/users/uat/model/")

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

tokenized_datasets = dataset.map(tokenize_function, batched=True)
small_train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(1000))
small_eval_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(1000))

model = AutoModelForSequenceClassification.from_pretrained("distilroberta-base", num_labels=5, cache_dir="/home/users/uat/model/")

training_args = TrainingArguments(output_dir="./test_trainer",
                                  overwrite_output_dir=True)
                                  
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=small_train_dataset,
    eval_dataset=small_eval_dataset,
)

trainer.train()

print("Done!")

