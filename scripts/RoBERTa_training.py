import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from transformers import (
    RobertaTokenizer,
    RobertaForSequenceClassification,
    Trainer,
    TrainingArguments
)
from torch.utils.data import Dataset


# ======================================================
# 1. LOAD MERGED DATA
# ======================================================
df = pd.read_csv("training_data.csv")

# Ensure no missing values
df = df.dropna(subset=["text", "ideology_score"])

print("Loaded dataset:", df.shape)
print(df.head())


# ======================================================
# 2. FIRM-LEVEL TRAIN/VAL/TEST SPLITS
#    (IMPORTANT: prevents model from memorizing firm names)
# ======================================================
firms = df["company"].unique()
np.random.shuffle(firms)

train_firms = firms[:45]
val_firms = firms[45:55]
test_firms = firms[55:]

train_df = df[df.company.isin(train_firms)]
val_df = df[df.company.isin(val_firms)]
test_df = df[df.company.isin(test_firms)]

print("\nSplit sizes:")
print("Train:", train_df.shape)
print("Val:", val_df.shape)
print("Test:", test_df.shape)


# ======================================================
# 3. TOKENIZER + CUSTOM DATASET CLASS
# ======================================================
model_name = "roberta-base"
tokenizer = RobertaTokenizer.from_pretrained(model_name)

MAX_LENGTH = 256  # You can increase to 512 if you have GPU memory


class IdeologyDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = texts.tolist()
        self.labels = labels.tolist()

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoded = tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=MAX_LENGTH,
            return_tensors="pt",
        )
        return {
            "input_ids": encoded["input_ids"].squeeze(),
            "attention_mask": encoded["attention_mask"].squeeze(),
            "labels": torch.tensor(self.labels[idx], dtype=torch.float)
        }


train_dataset = IdeologyDataset(train_df["text"], train_df["ideology_score"])
val_dataset = IdeologyDataset(val_df["text"], val_df["ideology_score"])
test_dataset = IdeologyDataset(test_df["text"], test_df["ideology_score"])


# ======================================================
# 4. LOAD ROBERTA MODEL FOR REGRESSION
# ======================================================
model = RobertaForSequenceClassification.from_pretrained(
    model_name,
    num_labels=1,                # *** REGRESSION ***
    problem_type="regression"   # ensures MSE loss
)


# ======================================================
# 5. TRAINING ARGUMENTS
# ======================================================
training_args = TrainingArguments(
    output_dir="./roberta_ideology_output",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=4,        # effective batch size 16
    learning_rate=2e-5,
    num_train_epochs=4,
    warmup_ratio=0.1,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=50,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss"
)


# ======================================================
# 6. DEFINE METRICS (MAE + MSE)
# ======================================================
def compute_metrics(eval_pred):
    preds, labels = eval_pred
    preds = preds.squeeze()
    mse = ((preds - labels) ** 2).mean()
    mae = (abs(preds - labels)).mean()
    return {"mse": mse, "mae": mae}


# ======================================================
# 7. INITIALIZE TRAINER
# ======================================================
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
)


# ======================================================
# 8. TRAIN THE MODEL
# ======================================================
trainer.train()


# ======================================================
# 9. EVALUATE ON TEST SET
# ======================================================
results = trainer.evaluate(test_dataset)
print("\nFinal Test Results:", results)

# Save model
trainer.save_model("./trained_roberta_ideology")
print("\nModel saved successfully.")
