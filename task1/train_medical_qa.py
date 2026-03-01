import subprocess
import sys

def install_packages():
    packages = [
        "transformers>=4.36.0",
        "datasets>=2.16.0",
        "peft>=0.7.0",
        "accelerate>=0.25.0",
        "evaluate>=0.4.0",
        "rouge-score>=0.1.2",
        "nltk>=3.8.0",
        "sentencepiece>=0.1.99",
        "protobuf>=4.25.0",
        "scikit-learn>=1.3.0",
        "matplotlib>=3.8.0",
        "seaborn>=0.13.0",
    ]
    for pkg in packages:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])
    print("All packages installed successfully.")

# install_packages()

import os
import time
import json
import random
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import nltk

from datasets import load_dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
)
from peft import LoraConfig, get_peft_model, TaskType
import evaluate

nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

CONFIG = {
    "model_name": "google/flan-t5-base",
    "dataset_name": "keivalya/MedQuad-MedicalQnA-5000",
    "max_input_length": 512,
    "max_target_length": 256,
    "train_split": 0.8,
    "val_split": 0.1,
    "test_split": 0.1,
    "batch_size": 8,
    "learning_rate": 3e-4,
    "num_epochs": 5,
    "lora_r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.1,
    "output_dir": "./results",
    "seed": SEED,
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")

dataset = load_dataset(CONFIG["dataset_name"])
print(f"\nDataset: {CONFIG['dataset_name']}, Total samples: {len(dataset['train'])}")
print(f"Sample Q: {dataset['train'][0]['Question'][:100]}...")
print(f"Sample A: {dataset['train'][0]['Answer'][:100]}...")

full_dataset = dataset["train"].shuffle(seed=SEED)
total_size = len(full_dataset)
train_size = int(total_size * CONFIG["train_split"])
val_size = int(total_size * CONFIG["val_split"])

split_dataset = DatasetDict({
    "train": full_dataset.select(range(train_size)),
    "validation": full_dataset.select(range(train_size, train_size + val_size)),
    "test": full_dataset.select(range(train_size + val_size, total_size)),
})

print(f"Train: {len(split_dataset['train'])}, Val: {len(split_dataset['validation'])}, Test: {len(split_dataset['test'])}")

tokenizer = AutoTokenizer.from_pretrained(CONFIG["model_name"])

def preprocess_function(examples):
    inputs = [f"Answer the following medical question: {q}" for q in examples["Question"]]
    targets = examples["Answer"]

    model_inputs = tokenizer(inputs, max_length=CONFIG["max_input_length"], truncation=True, padding="max_length")
    labels = tokenizer(targets, max_length=CONFIG["max_target_length"], truncation=True, padding="max_length")
    labels["input_ids"] = [
        [(l if l != tokenizer.pad_token_id else -100) for l in label]
        for label in labels["input_ids"]
    ]
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

print("Tokenizing datasets...")
tokenized_datasets = split_dataset.map(
    preprocess_function, batched=True,
    remove_columns=split_dataset["train"].column_names, desc="Tokenizing",
)
print("Tokenization complete.")

rouge_metric = evaluate.load("rouge")
bleu_metric = evaluate.load("bleu")

def compute_metrics(eval_preds):
    preds, labels = eval_preds
    preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    decoded_preds = [p.strip() for p in decoded_preds]
    decoded_labels = [l.strip() for l in decoded_labels]

    rouge_results = rouge_metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)

    bleu_scores = []
    for pred, ref in zip(decoded_preds, decoded_labels):
        try:
            pt, rt = pred.split(), ref.split()
            if pt and rt:
                bleu_scores.append(bleu_metric.compute(predictions=[pt], references=[[rt]])["bleu"])
            else:
                bleu_scores.append(0.0)
        except Exception:
            bleu_scores.append(0.0)

    return {k: round(v, 4) for k, v in {
        "rouge1": rouge_results["rouge1"], "rouge2": rouge_results["rouge2"],
        "rougeL": rouge_results["rougeL"], "bleu": np.mean(bleu_scores),
    }.items()}

print("Loading baseline model...")
baseline_model = AutoModelForSeq2SeqLM.from_pretrained(CONFIG["model_name"])
baseline_model.to(device)
model_params = sum(p.numel() for p in baseline_model.parameters())
print(f"Model parameters: {model_params:,} ({model_params/1e6:.1f}M)")

print("Evaluating baseline (zero-shot) on test set...")
baseline_start_time = time.time()
baseline_predictions, baseline_references, test_questions = [], [], []

baseline_model.eval()
with torch.no_grad():
    for i, sample in enumerate(split_dataset["test"]):
        input_text = f"Answer the following medical question: {sample['Question']}"
        inputs = tokenizer(input_text, return_tensors="pt", max_length=CONFIG["max_input_length"], truncation=True).to(device)
        outputs = baseline_model.generate(**inputs, max_new_tokens=CONFIG["max_target_length"], num_beams=4, early_stopping=True)
        baseline_predictions.append(tokenizer.decode(outputs[0], skip_special_tokens=True).strip())
        baseline_references.append(sample["Answer"].strip())
        test_questions.append(sample["Question"])
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(split_dataset['test'])} samples...")

baseline_eval_time = time.time() - baseline_start_time

baseline_rouge = rouge_metric.compute(predictions=baseline_predictions, references=baseline_references, use_stemmer=True)
baseline_bleu_scores = []
for pred, ref in zip(baseline_predictions, baseline_references):
    try:
        pt, rt = pred.split(), ref.split()
        if pt and rt:
            baseline_bleu_scores.append(bleu_metric.compute(predictions=[pt], references=[[rt]])["bleu"])
        else:
            baseline_bleu_scores.append(0.0)
    except Exception:
        baseline_bleu_scores.append(0.0)

baseline_metrics = {
    "rouge1": round(baseline_rouge["rouge1"], 4), "rouge2": round(baseline_rouge["rouge2"], 4),
    "rougeL": round(baseline_rouge["rougeL"], 4), "bleu": round(np.mean(baseline_bleu_scores), 4),
}

print(f"\nBaseline Results: ROUGE-1={baseline_metrics['rouge1']:.4f}, ROUGE-2={baseline_metrics['rouge2']:.4f}, ROUGE-L={baseline_metrics['rougeL']:.4f}, BLEU={baseline_metrics['bleu']:.4f}")

for i in range(min(3, len(test_questions))):
    print(f"\nQ: {test_questions[i][:150]}...")
    print(f"Baseline: {baseline_predictions[i][:150]}...")
    print(f"Reference: {baseline_references[i][:150]}...")

del baseline_model
torch.cuda.empty_cache() if torch.cuda.is_available() else None

print(f"""
Hardware Assessment:
  Full fine-tuning Flan-T5-Base (248M params) requires ~8-12 GB GPU memory.
  Available hardware: AMD Ryzen 5 5625U, no dedicated GPU - INFEASIBLE.
  Solution: LoRA reduces trainable parameters by ~99%, needing only ~2-3 GB.
  Running on Google Colab (T4 GPU, 15GB VRAM) for LoRA fine-tuning.
""")

print("Loading model for LoRA fine-tuning...")
model = AutoModelForSeq2SeqLM.from_pretrained(CONFIG["model_name"])

lora_config = LoraConfig(
    task_type=TaskType.SEQ_2_SEQ_LM,
    r=CONFIG["lora_r"], lora_alpha=CONFIG["lora_alpha"],
    lora_dropout=CONFIG["lora_dropout"],
    target_modules=["q", "v"], bias="none",
)

model = get_peft_model(model, lora_config)
model.to(device)

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
reduction = (1 - trainable_params / total_params) * 100

print(f"LoRA Config: r={CONFIG['lora_r']}, alpha={CONFIG['lora_alpha']}, targets=[q, v]")
print(f"Total params: {total_params:,}, Trainable: {trainable_params:,} ({trainable_params/total_params*100:.4f}%), Reduction: {reduction:.2f}%")

data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, padding=True)

training_args = Seq2SeqTrainingArguments(
    output_dir=CONFIG["output_dir"],
    num_train_epochs=CONFIG["num_epochs"],
    per_device_train_batch_size=CONFIG["batch_size"],
    per_device_eval_batch_size=CONFIG["batch_size"],
    learning_rate=CONFIG["learning_rate"],
    weight_decay=0.01, warmup_steps=100,
    eval_strategy="epoch", save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="rougeL", greater_is_better=True,
    predict_with_generate=True,
    generation_max_length=CONFIG["max_target_length"],
    logging_dir=f"{CONFIG['output_dir']}/logs", logging_steps=50,
    report_to="none", seed=SEED,
    fp16=torch.cuda.is_available(), dataloader_num_workers=0,
)

trainer = Seq2SeqTrainer(
    model=model, args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    tokenizer=tokenizer, data_collator=data_collator,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
)

print(f"\nTraining: {CONFIG['num_epochs']} epochs, batch_size={CONFIG['batch_size']}, lr={CONFIG['learning_rate']}, fp16={torch.cuda.is_available()}")
train_start_time = time.time()
train_result = trainer.train()
train_time = time.time() - train_start_time
print(f"Training complete: {train_time:.1f}s ({train_time/60:.1f} min), loss={train_result.training_loss:.4f}")

training_history = trainer.state.log_history
epoch_metrics = []
for entry in training_history:
    if "eval_rouge1" in entry:
        epoch_metrics.append({
            "epoch": entry.get("epoch", 0), "eval_loss": entry.get("eval_loss", 0),
            "eval_rouge1": entry.get("eval_rouge1", 0), "eval_rouge2": entry.get("eval_rouge2", 0),
            "eval_rougeL": entry.get("eval_rougeL", 0), "eval_bleu": entry.get("eval_bleu", 0),
        })

for em in epoch_metrics:
    print(f"  Epoch {em['epoch']:.0f}: Loss={em['eval_loss']:.4f}, R1={em['eval_rouge1']:.4f}, R2={em['eval_rouge2']:.4f}, RL={em['eval_rougeL']:.4f}, BLEU={em['eval_bleu']:.4f}")

print("\nEvaluating LoRA model on test set...")
lora_start_time = time.time()
lora_predictions, lora_references = [], []

model.eval()
with torch.no_grad():
    for i, sample in enumerate(split_dataset["test"]):
        input_text = f"Answer the following medical question: {sample['Question']}"
        inputs = tokenizer(input_text, return_tensors="pt", max_length=CONFIG["max_input_length"], truncation=True).to(device)
        outputs = model.generate(**inputs, max_new_tokens=CONFIG["max_target_length"], num_beams=4, early_stopping=True)
        lora_predictions.append(tokenizer.decode(outputs[0], skip_special_tokens=True).strip())
        lora_references.append(sample["Answer"].strip())
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(split_dataset['test'])} samples...")

lora_eval_time = time.time() - lora_start_time

lora_rouge = rouge_metric.compute(predictions=lora_predictions, references=lora_references, use_stemmer=True)
lora_bleu_scores = []
for pred, ref in zip(lora_predictions, lora_references):
    try:
        pt, rt = pred.split(), ref.split()
        if pt and rt:
            lora_bleu_scores.append(bleu_metric.compute(predictions=[pt], references=[[rt]])["bleu"])
        else:
            lora_bleu_scores.append(0.0)
    except Exception:
        lora_bleu_scores.append(0.0)

lora_metrics = {
    "rouge1": round(lora_rouge["rouge1"], 4), "rouge2": round(lora_rouge["rouge2"], 4),
    "rougeL": round(lora_rouge["rougeL"], 4), "bleu": round(np.mean(lora_bleu_scores), 4),
}

print(f"\nLoRA Results: ROUGE-1={lora_metrics['rouge1']:.4f}, ROUGE-2={lora_metrics['rouge2']:.4f}, ROUGE-L={lora_metrics['rougeL']:.4f}, BLEU={lora_metrics['bleu']:.4f}")

for i in range(min(3, len(test_questions))):
    print(f"\nQ: {test_questions[i][:150]}...")
    print(f"LoRA: {lora_predictions[i][:150]}...")
    print(f"Reference: {lora_references[i][:150]}...")

comparison_data = {
    "Metric": ["ROUGE-1", "ROUGE-2", "ROUGE-L", "BLEU"],
    "Baseline (Zero-Shot)": [baseline_metrics["rouge1"], baseline_metrics["rouge2"], baseline_metrics["rougeL"], baseline_metrics["bleu"]],
    "LoRA Fine-Tuned": [lora_metrics["rouge1"], lora_metrics["rouge2"], lora_metrics["rougeL"], lora_metrics["bleu"]],
}

comparison_df = pd.DataFrame(comparison_data)
comparison_df["Improvement"] = comparison_df["LoRA Fine-Tuned"] - comparison_df["Baseline (Zero-Shot)"]
comparison_df["Improvement (%)"] = ((comparison_df["Improvement"] / comparison_df["Baseline (Zero-Shot)"].replace(0, 1)) * 100).round(2)

print("\n" + comparison_df.to_string(index=False))
print(f"\nTotal params: {total_params:,}, Trainable (LoRA): {trainable_params:,}, Reduction: {reduction:.2f}%")
print(f"Baseline eval: {baseline_eval_time:.1f}s, LoRA eval: {lora_eval_time:.1f}s, Training: {train_time:.1f}s")

os.makedirs(CONFIG["output_dir"], exist_ok=True)

fig, ax = plt.subplots(figsize=(10, 6))
metrics_names = ["ROUGE-1", "ROUGE-2", "ROUGE-L", "BLEU"]
baseline_values = [baseline_metrics["rouge1"], baseline_metrics["rouge2"], baseline_metrics["rougeL"], baseline_metrics["bleu"]]
lora_values = [lora_metrics["rouge1"], lora_metrics["rouge2"], lora_metrics["rougeL"], lora_metrics["bleu"]]
x = np.arange(len(metrics_names))
width = 0.35
bars1 = ax.bar(x - width/2, baseline_values, width, label="Baseline (Zero-Shot)", color="#3498db", edgecolor="black", linewidth=0.5)
bars2 = ax.bar(x + width/2, lora_values, width, label="LoRA Fine-Tuned", color="#e74c3c", edgecolor="black", linewidth=0.5)
ax.set_xlabel("Evaluation Metric", fontsize=12)
ax.set_ylabel("Score", fontsize=12)
ax.set_title("Baseline vs LoRA Fine-Tuned: Medical Q&A Performance", fontsize=14)
ax.set_xticks(x)
ax.set_xticklabels(metrics_names, fontsize=11)
ax.legend(fontsize=11)
ax.grid(axis="y", alpha=0.3)
for bar in bars1:
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.005, f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=9)
for bar in bars2:
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.005, f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=9)
plt.tight_layout()
plt.savefig(f"{CONFIG['output_dir']}/metric_comparison.png", dpi=300, bbox_inches="tight")
plt.show()

if epoch_metrics:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    epochs = [em["epoch"] for em in epoch_metrics]
    axes[0].plot(epochs, [em["eval_loss"] for em in epoch_metrics], "b-o", linewidth=2, markersize=8, label="Validation Loss")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss"); axes[0].set_title("Validation Loss"); axes[0].legend(); axes[0].grid(alpha=0.3)
    axes[1].plot(epochs, [em["eval_rouge1"] for em in epoch_metrics], "r-o", linewidth=2, markersize=8, label="ROUGE-1")
    axes[1].plot(epochs, [em["eval_rouge2"] for em in epoch_metrics], "g-s", linewidth=2, markersize=8, label="ROUGE-2")
    axes[1].plot(epochs, [em["eval_rougeL"] for em in epoch_metrics], "b-^", linewidth=2, markersize=8, label="ROUGE-L")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Score"); axes[1].set_title("ROUGE Scores"); axes[1].legend(); axes[1].grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{CONFIG['output_dir']}/training_curves.png", dpi=300, bbox_inches="tight")
    plt.show()

fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.bar(["Total\nParameters", "Trainable\n(LoRA)"], [total_params, trainable_params], color=["#3498db", "#e74c3c"], edgecolor="black")
ax.set_ylabel("Parameters"); ax.set_title("Parameter Efficiency: LoRA vs Full Model"); ax.set_yscale("log"); ax.grid(axis="y", alpha=0.3)
for bar, val in zip(bars, [total_params, trainable_params]):
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() * 1.1, f"{val:,}", ha="center", va="bottom", fontsize=10)
plt.tight_layout()
plt.savefig(f"{CONFIG['output_dir']}/parameter_efficiency.png", dpi=300, bbox_inches="tight")
plt.show()

fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
N = len(metrics_names)
angles = [n / float(N) * 2 * np.pi for n in range(N)] + [0]
ax.plot(angles, baseline_values + baseline_values[:1], "o-", linewidth=2, label="Baseline", color="#3498db")
ax.fill(angles, baseline_values + baseline_values[:1], alpha=0.15, color="#3498db")
ax.plot(angles, lora_values + lora_values[:1], "o-", linewidth=2, label="LoRA Fine-Tuned", color="#e74c3c")
ax.fill(angles, lora_values + lora_values[:1], alpha=0.15, color="#e74c3c")
ax.set_xticks(angles[:-1]); ax.set_xticklabels(metrics_names, fontsize=12)
ax.set_title("Performance Radar: Baseline vs LoRA", fontsize=14, pad=20)
ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=11)
plt.tight_layout()
plt.savefig(f"{CONFIG['output_dir']}/radar_comparison.png", dpi=300, bbox_inches="tight")
plt.show()

results = {
    "config": CONFIG, "baseline_metrics": baseline_metrics, "lora_metrics": lora_metrics,
    "training_time_seconds": train_time, "baseline_eval_time_seconds": baseline_eval_time,
    "lora_eval_time_seconds": lora_eval_time, "total_parameters": total_params,
    "trainable_parameters": trainable_params, "parameter_reduction_percent": round(reduction, 2),
    "epoch_metrics": epoch_metrics,
    "hardware": {"device": str(device), "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A"},
}

with open(f"{CONFIG['output_dir']}/results.json", "w") as f:
    json.dump(results, f, indent=2)

comparison_df.to_csv(f"{CONFIG['output_dir']}/comparison.csv", index=False)
model.save_pretrained(f"{CONFIG['output_dir']}/lora_model")
tokenizer.save_pretrained(f"{CONFIG['output_dir']}/lora_model")

examples_df = pd.DataFrame({
    "Question": test_questions[:20], "Baseline_Prediction": baseline_predictions[:20],
    "LoRA_Prediction": lora_predictions[:20], "Reference": baseline_references[:20],
})
examples_df.to_csv(f"{CONFIG['output_dir']}/example_predictions.csv", index=False)

print(f"\nAll results saved to {CONFIG['output_dir']}/")
print(f"Model: {CONFIG['model_name']}, Dataset: {CONFIG['dataset_name']}")
print(f"LoRA: r={CONFIG['lora_r']}, alpha={CONFIG['lora_alpha']}, trainable={trainable_params:,} ({trainable_params/total_params*100:.4f}%)")
print(f"Training: {train_time/60:.1f} min on {device}")
