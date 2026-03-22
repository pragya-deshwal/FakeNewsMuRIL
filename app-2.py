import os
import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    trainer_utils
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support



torch.serialization.add_safe_globals([np.core.multiarray._reconstruct])
torch.serialization.add_safe_globals([np.ndarray])
torch.serialization.add_safe_globals([np.dtype]) 


os.environ["CUDA_VISIBLE_DEVICES"] = ""  
device = torch.device("cpu")

DATA_PATH = "" #choose your data path here

gujarati = pd.read_csv(DATA_PATH + "gujarati_news.csv", encoding="utf-8-sig")
hindi    = pd.read_csv(DATA_PATH + "hindi_news.csv", encoding="utf-8-sig")
marathi  = pd.read_csv(DATA_PATH + "marathi_news.csv", encoding="utf-8-sig")
telugu   = pd.read_csv(DATA_PATH + "telugu_news.csv", encoding="utf-8-sig")

df = pd.concat([gujarati, hindi, marathi, telugu], ignore_index=True)
df = df[['text', 'label']].dropna().reset_index(drop=True)
df = df.rename(columns={"label": "label"})
train_df, test_df = train_test_split(df, test_size=0.2, stratify=df['label'], random_state=42)


model_name = "google/muril-base-cased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
model.to(device)



def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True, max_length=128, padding="max_length")

train_dataset = Dataset.from_pandas(train_df)
test_dataset = Dataset.from_pandas(test_df)

tokenized_train = train_dataset.map(preprocess_function, batched=True, batch_size=100)
tokenized_test = test_dataset.map(preprocess_function, batched=True, batch_size=100)


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}


training_args = TrainingArguments(
    output_dir=DATA_PATH + "checkpoints",
    evaluation_strategy="steps",
    eval_steps=1000,
    save_strategy="steps",
    save_steps=1000,
    save_total_limit=2,
    logging_dir=DATA_PATH + "logs",
    logging_steps=50,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    num_train_epochs=3,
    no_cuda=True,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy"
)


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_test,
    tokenizer=tokenizer,
    data_collator=DataCollatorWithPadding(tokenizer),
    compute_metrics=compute_metrics
)

resume_ckpt = os.path.normpath(os.path.join(training_args.output_dir, "checkpoint-22000"))

def is_valid_checkpoint(path):
    required = ["trainer_state.json", "optimizer.pt", "scheduler.pt"]
    return os.path.isdir(path) and all(os.path.isfile(os.path.join(path, f)) for f in required)


if is_valid_checkpoint(resume_ckpt):
    print(" Resuming from:", resume_ckpt.replace("\\", "/"))
    
    def skip_rng_load(*args, **kwargs):
        print(" Skipping RNG state loading")
    trainer._load_rng_state = skip_rng_load

    trainer.train(resume_from_checkpoint=resume_ckpt)
else:
    print(" Starting fresh training")
    trainer.train()






trainer.save_model(DATA_PATH + "final_muril_model")
tokenizer.save_pretrained(DATA_PATH + "final_muril_model")


def predict_fake_news(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128, padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    outputs = model(**inputs)
    probs = torch.softmax(outputs.logits, dim=1)
    pred = torch.argmax(probs, dim=1).item()
    label = "Fake" if pred == 1 else "Real"
    confidence = probs[0][pred].item()
    return label, confidence

#  EXAMPLE PREDICTIONS
examples = [
    "મોદી સરકાર નવી યોજના લાવી છે",
    "सरकार ने योजना को शुरू किया",
    "सरकारने योजना सुरू केली आहे",
    "ప్రభుత్వం కొత్త పథకాన్ని ప్రారంభించింది"
]

for text in examples:
    label, confidence = predict_fake_news(text)
    print(f"\n Text: {text}\n Prediction: {label} (Confidence: {confidence*100:.2f}%)")
