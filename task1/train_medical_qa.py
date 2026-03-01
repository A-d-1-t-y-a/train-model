"""
Medical Question Answering with Flan-T5-Base: Baseline vs LoRA Fine-Tuning
===========================================================================
Module: 7043SCN - Generative AI and Reinforcement Learning
Task 1: Comparative LLM Fine-Tuning Project

This script fine-tunes Google's Flan-T5-Base model for Medical Question Answering
using LoRA (Low-Rank Adaptation) and compares it against the zero-shot baseline.

Designed to run on Google Colab (free tier T4 GPU).

Hardware Justification:
    Full fine-tuning Flan-T5-Base (250M params) requires ~4GB for model weights
    + ~8GB for optimizer states (AdamW) + ~4GB for gradients = ~16GB minimum.
    Available hardware (AMD Ryzen 5 5625U, no dedicated GPU) is insufficient
    for full fine-tuning. Therefore, we use LoRA for parameter-efficient
    fine-tuning, which reduces trainable parameters by ~99%.

Usage (Google Colab):
    1. Upload this script to Colab or copy cells into a notebook
    2. Ensure GPU runtime is enabled (Runtime > Change runtime type > T4 GPU)
    3. Run all cells sequentially

Author: [Student Name]
Date: March 2026
"""

# ============================================================================
# CELL 1: Environment Setup and Installation
# ============================================================================

import subprocess
import sys

def install_packages():
    """Install required packages for the project."""
    packages = [
        "transformers>=4.36.0",
        "datasets>=2.16.0",
        "peft>=0.7.0",
        "accelerate>=0.25.0",
        "evaluate>=0.4.0",
        "rouge-score>=0.1.2",
        "nltk>=3.8.0",
        "bitsandbytes>=0.41.0",
        "sentencepiece>=0.1.99",
        "protobuf>=4.25.0",
        "scikit-learn>=1.3.0",
        "matplotlib>=3.8.0",
        "seaborn>=0.13.0",
    ]
    for pkg in packages:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])
    print("All packages installed successfully.")

# Uncomment the line below when running in Colab:
# install_packages()

# ============================================================================
# CELL 2: Imports and Configuration
# ============================================================================

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
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType,
    prepare_model_for_kbit_training,
)
import evaluate

# Download NLTK data for BLEU
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)

# Set seeds for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# Configuration
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

# Check device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")

# ============================================================================
# CELL 3: Dataset Loading and Preprocessing
# ============================================================================

print("=" * 60)
print("STEP 1: Loading and Preparing Dataset")
print("=" * 60)

# Load the MedQuAD Medical QA dataset
dataset = load_dataset(CONFIG["dataset_name"])
print(f"\nDataset loaded: {CONFIG['dataset_name']}")
print(f"Total samples: {len(dataset['train'])}")
print(f"\nSample entry:")
print(f"  Question: {dataset['train'][0]['Question'][:100]}...")
print(f"  Answer: {dataset['train'][0]['Answer'][:100]}...")

# Create train/val/test splits
full_dataset = dataset["train"].shuffle(seed=SEED)
total_size = len(full_dataset)
train_size = int(total_size * CONFIG["train_split"])
val_size = int(total_size * CONFIG["val_split"])
test_size = total_size - train_size - val_size

split_dataset = DatasetDict({
    "train": full_dataset.select(range(train_size)),
    "validation": full_dataset.select(range(train_size, train_size + val_size)),
    "test": full_dataset.select(range(train_size + val_size, total_size)),
})

print(f"\nData splits:")
print(f"  Train: {len(split_dataset['train'])} samples ({CONFIG['train_split']*100:.0f}%)")
print(f"  Validation: {len(split_dataset['validation'])} samples ({CONFIG['val_split']*100:.0f}%)")
print(f"  Test: {len(split_dataset['test'])} samples ({CONFIG['test_split']*100:.0f}%)")

# ============================================================================
# CELL 4: Tokenizer and Preprocessing
# ============================================================================

print("\n" + "=" * 60)
print("STEP 2: Tokenization and Preprocessing")
print("=" * 60)

tokenizer = AutoTokenizer.from_pretrained(CONFIG["model_name"])

def preprocess_function(examples):
    """
    Format medical Q&A pairs for Flan-T5.
    Input format: "Answer the following medical question: {question}"
    Target format: "{answer}"
    """
    inputs = [
        f"Answer the following medical question: {q}"
        for q in examples["Question"]
    ]
    targets = examples["Answer"]

    model_inputs = tokenizer(
        inputs,
        max_length=CONFIG["max_input_length"],
        truncation=True,
        padding="max_length",
    )

    labels = tokenizer(
        targets,
        max_length=CONFIG["max_target_length"],
        truncation=True,
        padding="max_length",
    )

    # Replace padding token id with -100 so it's ignored in loss
    labels["input_ids"] = [
        [(l if l != tokenizer.pad_token_id else -100) for l in label]
        for label in labels["input_ids"]
    ]

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# Tokenize all splits
print("Tokenizing datasets...")
tokenized_datasets = split_dataset.map(
    preprocess_function,
    batched=True,
    remove_columns=split_dataset["train"].column_names,
    desc="Tokenizing",
)

print(f"Tokenization complete.")
print(f"  Input max length: {CONFIG['max_input_length']}")
print(f"  Target max length: {CONFIG['max_target_length']}")

# ============================================================================
# CELL 5: Evaluation Metrics Setup
# ============================================================================

print("\n" + "=" * 60)
print("STEP 3: Setting Up Evaluation Metrics")
print("=" * 60)

rouge_metric = evaluate.load("rouge")
bleu_metric = evaluate.load("bleu")

def compute_metrics(eval_preds):
    """Compute ROUGE and BLEU metrics for model evaluation."""
    preds, labels = eval_preds

    # Decode predictions
    preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

    # Decode labels
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Strip whitespace
    decoded_preds = [pred.strip() for pred in decoded_preds]
    decoded_labels = [label.strip() for label in decoded_labels]

    # Compute ROUGE
    rouge_results = rouge_metric.compute(
        predictions=decoded_preds,
        references=decoded_labels,
        use_stemmer=True,
    )

    # Compute BLEU (sentence-level average)
    bleu_scores = []
    for pred, ref in zip(decoded_preds, decoded_labels):
        try:
            pred_tokens = pred.split()
            ref_tokens = ref.split()
            if len(pred_tokens) > 0 and len(ref_tokens) > 0:
                score = bleu_metric.compute(
                    predictions=[pred_tokens],
                    references=[[ref_tokens]],
                )
                bleu_scores.append(score["bleu"])
            else:
                bleu_scores.append(0.0)
        except Exception:
            bleu_scores.append(0.0)

    results = {
        "rouge1": rouge_results["rouge1"],
        "rouge2": rouge_results["rouge2"],
        "rougeL": rouge_results["rougeL"],
        "bleu": np.mean(bleu_scores),
    }

    return {k: round(v, 4) for k, v in results.items()}

print("Metrics configured: ROUGE-1, ROUGE-2, ROUGE-L, BLEU")

# ============================================================================
# CELL 6: Baseline Model Evaluation (Zero-Shot)
# ============================================================================

print("\n" + "=" * 60)
print("STEP 4: Baseline Model Evaluation (Zero-Shot)")
print("=" * 60)

print(f"\nLoading pre-trained model: {CONFIG['model_name']}...")
baseline_model = AutoModelForSeq2SeqLM.from_pretrained(CONFIG["model_name"])
baseline_model.to(device)

model_params = sum(p.numel() for p in baseline_model.parameters())
print(f"Model parameters: {model_params:,} ({model_params/1e6:.1f}M)")

# Evaluate baseline on test set
print("\nEvaluating baseline model on test set...")
baseline_start_time = time.time()

baseline_predictions = []
baseline_references = []
test_questions = []

baseline_model.eval()
with torch.no_grad():
    for i, sample in enumerate(split_dataset["test"]):
        question = sample["Question"]
        reference = sample["Answer"]

        input_text = f"Answer the following medical question: {question}"
        inputs = tokenizer(
            input_text,
            return_tensors="pt",
            max_length=CONFIG["max_input_length"],
            truncation=True,
        ).to(device)

        outputs = baseline_model.generate(
            **inputs,
            max_new_tokens=CONFIG["max_target_length"],
            num_beams=4,
            early_stopping=True,
        )

        prediction = tokenizer.decode(outputs[0], skip_special_tokens=True)
        baseline_predictions.append(prediction.strip())
        baseline_references.append(reference.strip())
        test_questions.append(question)

        if (i + 1) % 50 == 0:
            print(f"  Processed {i+1}/{len(split_dataset['test'])} samples...")

baseline_eval_time = time.time() - baseline_start_time

# Compute baseline metrics
baseline_rouge = rouge_metric.compute(
    predictions=baseline_predictions,
    references=baseline_references,
    use_stemmer=True,
)

baseline_bleu_scores = []
for pred, ref in zip(baseline_predictions, baseline_references):
    try:
        pred_tokens = pred.split()
        ref_tokens = ref.split()
        if len(pred_tokens) > 0 and len(ref_tokens) > 0:
            score = bleu_metric.compute(
                predictions=[pred_tokens],
                references=[[ref_tokens]],
            )
            baseline_bleu_scores.append(score["bleu"])
        else:
            baseline_bleu_scores.append(0.0)
    except Exception:
        baseline_bleu_scores.append(0.0)

baseline_metrics = {
    "rouge1": round(baseline_rouge["rouge1"], 4),
    "rouge2": round(baseline_rouge["rouge2"], 4),
    "rougeL": round(baseline_rouge["rougeL"], 4),
    "bleu": round(np.mean(baseline_bleu_scores), 4),
}

print(f"\n--- Baseline Results (Zero-Shot) ---")
print(f"  ROUGE-1: {baseline_metrics['rouge1']:.4f}")
print(f"  ROUGE-2: {baseline_metrics['rouge2']:.4f}")
print(f"  ROUGE-L: {baseline_metrics['rougeL']:.4f}")
print(f"  BLEU:    {baseline_metrics['bleu']:.4f}")
print(f"  Evaluation time: {baseline_eval_time:.1f}s")

# Show example predictions
print(f"\n--- Example Baseline Predictions ---")
for i in range(min(3, len(test_questions))):
    print(f"\nQ: {test_questions[i][:150]}...")
    print(f"Baseline: {baseline_predictions[i][:150]}...")
    print(f"Reference: {baseline_references[i][:150]}...")

# Clean up baseline model to free memory
del baseline_model
torch.cuda.empty_cache() if torch.cuda.is_available() else None

# ============================================================================
# CELL 7: Hardware Feasibility Assessment
# ============================================================================

print("\n" + "=" * 60)
print("STEP 5: Hardware Feasibility Assessment for Full Fine-Tuning")
print("=" * 60)

print(f"""
Hardware Assessment:
--------------------
Full fine-tuning Flan-T5-Base (248M parameters) requires:
  - Model weights (FP32):  ~1.0 GB
  - Optimizer states (AdamW, 2x model size): ~2.0 GB
  - Gradients: ~1.0 GB
  - Activations (batch_size=8, seq_len=512): ~4-8 GB
  - Total estimated: ~8-12 GB GPU memory

Available hardware: AMD Ryzen 5 5625U with integrated Radeon Graphics (no dedicated GPU).
Integrated GPU memory is shared with system RAM and is insufficient for deep learning training.

Conclusion: Full fine-tuning is INFEASIBLE on available hardware.
Therefore, we use LoRA (Low-Rank Adaptation) for parameter-efficient fine-tuning,
which reduces trainable parameters by ~99%, requiring only ~2-3 GB GPU memory.

Note: This script is designed to run on Google Colab (free T4 GPU, 15GB VRAM),
which is sufficient for LoRA fine-tuning but would be tight for full fine-tuning
of larger models.
""")

# ============================================================================
# CELL 8: LoRA Fine-Tuning Setup
# ============================================================================

print("=" * 60)
print("STEP 6: LoRA Fine-Tuning Configuration")
print("=" * 60)

# Load fresh model for fine-tuning
print(f"\nLoading model for LoRA fine-tuning...")
model = AutoModelForSeq2SeqLM.from_pretrained(CONFIG["model_name"])

# Configure LoRA
lora_config = LoraConfig(
    task_type=TaskType.SEQ_2_SEQ_LM,
    r=CONFIG["lora_r"],
    lora_alpha=CONFIG["lora_alpha"],
    lora_dropout=CONFIG["lora_dropout"],
    target_modules=["q", "v"],  # Apply LoRA to query and value projections
    bias="none",
)

# Apply LoRA to the model
model = get_peft_model(model, lora_config)
model.to(device)

# Print parameter comparison
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
reduction = (1 - trainable_params / total_params) * 100

print(f"\n--- LoRA Configuration ---")
print(f"  Rank (r): {CONFIG['lora_r']}")
print(f"  Alpha: {CONFIG['lora_alpha']}")
print(f"  Dropout: {CONFIG['lora_dropout']}")
print(f"  Target modules: q, v (query and value projections)")
print(f"\n--- Parameter Efficiency ---")
print(f"  Total parameters:     {total_params:>12,}")
print(f"  Trainable parameters: {trainable_params:>12,}")
print(f"  Parameter reduction:  {reduction:.2f}%")
print(f"  Trainable ratio:      {trainable_params/total_params*100:.4f}%")

# ============================================================================
# CELL 9: Training
# ============================================================================

print("\n" + "=" * 60)
print("STEP 7: LoRA Fine-Tuning Training")
print("=" * 60)

# Data collator
data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    model=model,
    padding=True,
)

# Training arguments
training_args = Seq2SeqTrainingArguments(
    output_dir=CONFIG["output_dir"],
    num_train_epochs=CONFIG["num_epochs"],
    per_device_train_batch_size=CONFIG["batch_size"],
    per_device_eval_batch_size=CONFIG["batch_size"],
    learning_rate=CONFIG["learning_rate"],
    weight_decay=0.01,
    warmup_steps=100,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="rougeL",
    greater_is_better=True,
    predict_with_generate=True,
    generation_max_length=CONFIG["max_target_length"],
    logging_dir=f"{CONFIG['output_dir']}/logs",
    logging_steps=50,
    report_to="none",
    seed=SEED,
    fp16=torch.cuda.is_available(),
    dataloader_num_workers=0,
)

# Trainer
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
)

print(f"\nTraining configuration:")
print(f"  Epochs: {CONFIG['num_epochs']}")
print(f"  Batch size: {CONFIG['batch_size']}")
print(f"  Learning rate: {CONFIG['learning_rate']}")
print(f"  Warmup steps: 100")
print(f"  Weight decay: 0.01")
print(f"  FP16: {torch.cuda.is_available()}")

print(f"\nStarting training...")
train_start_time = time.time()
train_result = trainer.train()
train_time = time.time() - train_start_time

print(f"\n--- Training Complete ---")
print(f"  Training time: {train_time:.1f}s ({train_time/60:.1f} min)")
print(f"  Final training loss: {train_result.training_loss:.4f}")

# Save training metrics per epoch
training_history = trainer.state.log_history
epoch_metrics = []
for entry in training_history:
    if "eval_rouge1" in entry:
        epoch_metrics.append({
            "epoch": entry.get("epoch", 0),
            "eval_loss": entry.get("eval_loss", 0),
            "eval_rouge1": entry.get("eval_rouge1", 0),
            "eval_rouge2": entry.get("eval_rouge2", 0),
            "eval_rougeL": entry.get("eval_rougeL", 0),
            "eval_bleu": entry.get("eval_bleu", 0),
        })

if epoch_metrics:
    print(f"\n--- Training Progress (Validation Metrics Per Epoch) ---")
    for em in epoch_metrics:
        print(f"  Epoch {em['epoch']:.0f}: Loss={em['eval_loss']:.4f}, "
              f"R1={em['eval_rouge1']:.4f}, R2={em['eval_rouge2']:.4f}, "
              f"RL={em['eval_rougeL']:.4f}, BLEU={em['eval_bleu']:.4f}")

# ============================================================================
# CELL 10: Fine-Tuned Model Evaluation
# ============================================================================

print("\n" + "=" * 60)
print("STEP 8: Fine-Tuned Model Evaluation on Test Set")
print("=" * 60)

print("\nEvaluating LoRA fine-tuned model on test set...")
lora_start_time = time.time()

lora_predictions = []
lora_references = []

model.eval()
with torch.no_grad():
    for i, sample in enumerate(split_dataset["test"]):
        question = sample["Question"]
        reference = sample["Answer"]

        input_text = f"Answer the following medical question: {question}"
        inputs = tokenizer(
            input_text,
            return_tensors="pt",
            max_length=CONFIG["max_input_length"],
            truncation=True,
        ).to(device)

        outputs = model.generate(
            **inputs,
            max_new_tokens=CONFIG["max_target_length"],
            num_beams=4,
            early_stopping=True,
        )

        prediction = tokenizer.decode(outputs[0], skip_special_tokens=True)
        lora_predictions.append(prediction.strip())
        lora_references.append(reference.strip())

        if (i + 1) % 50 == 0:
            print(f"  Processed {i+1}/{len(split_dataset['test'])} samples...")

lora_eval_time = time.time() - lora_start_time

# Compute LoRA metrics
lora_rouge = rouge_metric.compute(
    predictions=lora_predictions,
    references=lora_references,
    use_stemmer=True,
)

lora_bleu_scores = []
for pred, ref in zip(lora_predictions, lora_references):
    try:
        pred_tokens = pred.split()
        ref_tokens = ref.split()
        if len(pred_tokens) > 0 and len(ref_tokens) > 0:
            score = bleu_metric.compute(
                predictions=[pred_tokens],
                references=[[ref_tokens]],
            )
            lora_bleu_scores.append(score["bleu"])
        else:
            lora_bleu_scores.append(0.0)
    except Exception:
        lora_bleu_scores.append(0.0)

lora_metrics = {
    "rouge1": round(lora_rouge["rouge1"], 4),
    "rouge2": round(lora_rouge["rouge2"], 4),
    "rougeL": round(lora_rouge["rougeL"], 4),
    "bleu": round(np.mean(lora_bleu_scores), 4),
}

print(f"\n--- LoRA Fine-Tuned Results ---")
print(f"  ROUGE-1: {lora_metrics['rouge1']:.4f}")
print(f"  ROUGE-2: {lora_metrics['rouge2']:.4f}")
print(f"  ROUGE-L: {lora_metrics['rougeL']:.4f}")
print(f"  BLEU:    {lora_metrics['bleu']:.4f}")
print(f"  Evaluation time: {lora_eval_time:.1f}s")

# Show example predictions
print(f"\n--- Example LoRA Fine-Tuned Predictions ---")
for i in range(min(3, len(test_questions))):
    print(f"\nQ: {test_questions[i][:150]}...")
    print(f"LoRA: {lora_predictions[i][:150]}...")
    print(f"Reference: {lora_references[i][:150]}...")

# ============================================================================
# CELL 11: Comparative Analysis
# ============================================================================

print("\n" + "=" * 60)
print("STEP 9: Comparative Analysis")
print("=" * 60)

# Create comparison table
comparison_data = {
    "Metric": ["ROUGE-1", "ROUGE-2", "ROUGE-L", "BLEU"],
    "Baseline (Zero-Shot)": [
        baseline_metrics["rouge1"],
        baseline_metrics["rouge2"],
        baseline_metrics["rougeL"],
        baseline_metrics["bleu"],
    ],
    "LoRA Fine-Tuned": [
        lora_metrics["rouge1"],
        lora_metrics["rouge2"],
        lora_metrics["rougeL"],
        lora_metrics["bleu"],
    ],
}

comparison_df = pd.DataFrame(comparison_data)
comparison_df["Improvement"] = comparison_df["LoRA Fine-Tuned"] - comparison_df["Baseline (Zero-Shot)"]
comparison_df["Improvement (%)"] = (
    (comparison_df["Improvement"] / comparison_df["Baseline (Zero-Shot)"].replace(0, 1)) * 100
).round(2)

print("\n--- Quantitative Comparison ---")
print(comparison_df.to_string(index=False))

# Resource usage comparison
print(f"\n--- Resource Usage ---")
print(f"  {'Metric':<30} {'Baseline':>15} {'LoRA':>15}")
print(f"  {'-'*60}")
print(f"  {'Total Parameters':<30} {model_params:>15,} {total_params:>15,}")
print(f"  {'Trainable Parameters':<30} {'N/A (zero-shot)':>15} {trainable_params:>15,}")
print(f"  {'Parameter Reduction':<30} {'N/A':>15} {reduction:>14.2f}%")
print(f"  {'Evaluation Time (s)':<30} {baseline_eval_time:>15.1f} {lora_eval_time:>15.1f}")
print(f"  {'Training Time (s)':<30} {'N/A':>15} {train_time:>15.1f}")

# ============================================================================
# CELL 12: Visualization
# ============================================================================

print("\n" + "=" * 60)
print("STEP 10: Generating Visualizations")
print("=" * 60)

os.makedirs(CONFIG["output_dir"], exist_ok=True)

# Plot 1: Metric Comparison Bar Chart
fig, ax = plt.subplots(figsize=(10, 6))
metrics_names = ["ROUGE-1", "ROUGE-2", "ROUGE-L", "BLEU"]
baseline_values = [baseline_metrics["rouge1"], baseline_metrics["rouge2"],
                   baseline_metrics["rougeL"], baseline_metrics["bleu"]]
lora_values = [lora_metrics["rouge1"], lora_metrics["rouge2"],
               lora_metrics["rougeL"], lora_metrics["bleu"]]

x = np.arange(len(metrics_names))
width = 0.35

bars1 = ax.bar(x - width/2, baseline_values, width, label="Baseline (Zero-Shot)",
               color="#3498db", edgecolor="black", linewidth=0.5)
bars2 = ax.bar(x + width/2, lora_values, width, label="LoRA Fine-Tuned",
               color="#e74c3c", edgecolor="black", linewidth=0.5)

ax.set_xlabel("Evaluation Metric", fontsize=12)
ax.set_ylabel("Score", fontsize=12)
ax.set_title("Baseline vs LoRA Fine-Tuned: Medical Q&A Performance", fontsize=14)
ax.set_xticks(x)
ax.set_xticklabels(metrics_names, fontsize=11)
ax.legend(fontsize=11)
ax.grid(axis="y", alpha=0.3)

# Add value labels on bars
for bar in bars1:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 0.005,
            f"{height:.3f}", ha="center", va="bottom", fontsize=9)
for bar in bars2:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 0.005,
            f"{height:.3f}", ha="center", va="bottom", fontsize=9)

plt.tight_layout()
plt.savefig(f"{CONFIG['output_dir']}/metric_comparison.png", dpi=300, bbox_inches="tight")
plt.show()
print("  Saved: metric_comparison.png")

# Plot 2: Training Loss Curve
if epoch_metrics:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    epochs = [em["epoch"] for em in epoch_metrics]

    # Loss curve
    axes[0].plot(epochs, [em["eval_loss"] for em in epoch_metrics],
                 "b-o", linewidth=2, markersize=8, label="Validation Loss")
    axes[0].set_xlabel("Epoch", fontsize=12)
    axes[0].set_ylabel("Loss", fontsize=12)
    axes[0].set_title("Training Progress: Validation Loss", fontsize=13)
    axes[0].legend(fontsize=11)
    axes[0].grid(alpha=0.3)

    # ROUGE scores over epochs
    axes[1].plot(epochs, [em["eval_rouge1"] for em in epoch_metrics],
                 "r-o", linewidth=2, markersize=8, label="ROUGE-1")
    axes[1].plot(epochs, [em["eval_rouge2"] for em in epoch_metrics],
                 "g-s", linewidth=2, markersize=8, label="ROUGE-2")
    axes[1].plot(epochs, [em["eval_rougeL"] for em in epoch_metrics],
                 "b-^", linewidth=2, markersize=8, label="ROUGE-L")
    axes[1].set_xlabel("Epoch", fontsize=12)
    axes[1].set_ylabel("Score", fontsize=12)
    axes[1].set_title("Training Progress: ROUGE Scores", fontsize=13)
    axes[1].legend(fontsize=11)
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{CONFIG['output_dir']}/training_curves.png", dpi=300, bbox_inches="tight")
    plt.show()
    print("  Saved: training_curves.png")

# Plot 3: Parameter Efficiency Visualization
fig, ax = plt.subplots(figsize=(8, 5))
categories = ["Total\nParameters", "Trainable\nParameters\n(LoRA)"]
values = [total_params, trainable_params]
colors = ["#3498db", "#e74c3c"]

bars = ax.bar(categories, values, color=colors, edgecolor="black", linewidth=0.5)
ax.set_ylabel("Number of Parameters", fontsize=12)
ax.set_title("Parameter Efficiency: LoRA vs Full Model", fontsize=14)
ax.set_yscale("log")
ax.grid(axis="y", alpha=0.3)

for bar, val in zip(bars, values):
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() * 1.1,
            f"{val:,}", ha="center", va="bottom", fontsize=10)

plt.tight_layout()
plt.savefig(f"{CONFIG['output_dir']}/parameter_efficiency.png", dpi=300, bbox_inches="tight")
plt.show()
print("  Saved: parameter_efficiency.png")

# Plot 4: Improvement Radar Chart
fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
categories_radar = metrics_names
N = len(categories_radar)
angles = [n / float(N) * 2 * np.pi for n in range(N)]
angles += angles[:1]

baseline_radar = baseline_values + baseline_values[:1]
lora_radar = lora_values + lora_values[:1]

ax.plot(angles, baseline_radar, "o-", linewidth=2, label="Baseline", color="#3498db")
ax.fill(angles, baseline_radar, alpha=0.15, color="#3498db")
ax.plot(angles, lora_radar, "o-", linewidth=2, label="LoRA Fine-Tuned", color="#e74c3c")
ax.fill(angles, lora_radar, alpha=0.15, color="#e74c3c")

ax.set_xticks(angles[:-1])
ax.set_xticklabels(categories_radar, fontsize=12)
ax.set_title("Performance Radar: Baseline vs LoRA", fontsize=14, pad=20)
ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=11)

plt.tight_layout()
plt.savefig(f"{CONFIG['output_dir']}/radar_comparison.png", dpi=300, bbox_inches="tight")
plt.show()
print("  Saved: radar_comparison.png")

# ============================================================================
# CELL 13: Save Results
# ============================================================================

print("\n" + "=" * 60)
print("STEP 11: Saving Results")
print("=" * 60)

# Save all results to JSON
results = {
    "config": CONFIG,
    "baseline_metrics": baseline_metrics,
    "lora_metrics": lora_metrics,
    "training_time_seconds": train_time,
    "baseline_eval_time_seconds": baseline_eval_time,
    "lora_eval_time_seconds": lora_eval_time,
    "total_parameters": total_params,
    "trainable_parameters": trainable_params,
    "parameter_reduction_percent": round(reduction, 2),
    "epoch_metrics": epoch_metrics,
    "hardware": {
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A",
    },
}

with open(f"{CONFIG['output_dir']}/results.json", "w") as f:
    json.dump(results, f, indent=2)
print(f"  Results saved to {CONFIG['output_dir']}/results.json")

# Save comparison table
comparison_df.to_csv(f"{CONFIG['output_dir']}/comparison.csv", index=False)
print(f"  Comparison table saved to {CONFIG['output_dir']}/comparison.csv")

# Save model
model.save_pretrained(f"{CONFIG['output_dir']}/lora_model")
tokenizer.save_pretrained(f"{CONFIG['output_dir']}/lora_model")
print(f"  LoRA model saved to {CONFIG['output_dir']}/lora_model/")

# Save example predictions
examples_df = pd.DataFrame({
    "Question": test_questions[:20],
    "Baseline_Prediction": baseline_predictions[:20],
    "LoRA_Prediction": lora_predictions[:20],
    "Reference": baseline_references[:20],
})
examples_df.to_csv(f"{CONFIG['output_dir']}/example_predictions.csv", index=False)
print(f"  Example predictions saved to {CONFIG['output_dir']}/example_predictions.csv")

# ============================================================================
# CELL 14: Summary
# ============================================================================

print("\n" + "=" * 60)
print("FINAL SUMMARY")
print("=" * 60)

print(f"""
Project: Medical Question Answering with Flan-T5-Base
=====================================================

Model: {CONFIG['model_name']} (248M parameters)
Dataset: {CONFIG['dataset_name']}
Task: Medical Question Answering (Generative)
Fine-Tuning Method: LoRA (r={CONFIG['lora_r']}, alpha={CONFIG['lora_alpha']})

Results Summary:
                    Baseline    LoRA      Improvement
  ROUGE-1:          {baseline_metrics['rouge1']:.4f}      {lora_metrics['rouge1']:.4f}    {lora_metrics['rouge1']-baseline_metrics['rouge1']:+.4f}
  ROUGE-2:          {baseline_metrics['rouge2']:.4f}      {lora_metrics['rouge2']:.4f}    {lora_metrics['rouge2']-baseline_metrics['rouge2']:+.4f}
  ROUGE-L:          {baseline_metrics['rougeL']:.4f}      {lora_metrics['rougeL']:.4f}    {lora_metrics['rougeL']-baseline_metrics['rougeL']:+.4f}
  BLEU:             {baseline_metrics['bleu']:.4f}      {lora_metrics['bleu']:.4f}    {lora_metrics['bleu']-baseline_metrics['bleu']:+.4f}

Key Findings:
  - Total parameters: {total_params:,}
  - Trainable parameters (LoRA): {trainable_params:,} ({trainable_params/total_params*100:.4f}%)
  - Training time: {train_time/60:.1f} minutes
  - Hardware: {str(device)}

Files generated:
  - {CONFIG['output_dir']}/metric_comparison.png
  - {CONFIG['output_dir']}/training_curves.png
  - {CONFIG['output_dir']}/parameter_efficiency.png
  - {CONFIG['output_dir']}/radar_comparison.png
  - {CONFIG['output_dir']}/results.json
  - {CONFIG['output_dir']}/comparison.csv
  - {CONFIG['output_dir']}/example_predictions.csv
  - {CONFIG['output_dir']}/lora_model/
""")

print("=" * 60)
print("Script complete. All results saved.")
print("=" * 60)
