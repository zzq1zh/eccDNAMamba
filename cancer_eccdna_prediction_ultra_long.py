# =========================================
# Minimal Inference Script
# =========================================

import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from datasets import load_dataset
from transformers import (
    AutoModel,
    AutoTokenizer,
    AutoConfig,
    PreTrainedModel,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer
)
from BiMambaForMaskedLM import BiMambaForMaskedLM
from ig import integrated_gradients_margin, plot_heatmap_1d
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# ======================
# Path Configuration
# ======================
DATASET_NAME  = "eccDNAMamba/cancer_eccdna_prediction_ultra_long"
FINETUNED_PATH = "eccDNAMamba/eccDNAMamba_1M_cancer_eccdna_prediction_ultra_long" 

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", DEVICE)

# ======================
# Tokenizer + circular augmentation
# ======================
tokenizer = AutoTokenizer.from_pretrained(FINETUNED_PATH, trust_remote_code=True)

def circular_augmentation(seq, extend_len=64):
    """Concatenate the first `extend_len` bases to form a circularly augmented sequence"""
    tokens = tokenizer(seq, add_special_tokens=True, truncation=False)["input_ids"]
    circular_aug = tokens + tokens[1:extend_len+1]
    return {"input_ids": circular_aug}

# ======================
# Model Definition
# ======================
class BiMambaForClassification(PreTrainedModel):
    def __init__(self, config, num_classes=2, freeze_except_last_k=2):
        super().__init__(config)
        self.backbone = BiMambaForMaskedLM(config)
        self.config = config
        hidden_size = config.d_model
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, num_classes)
        )

    def masked_mean_pooling(self, hidden, attention_mask):
        mask = attention_mask.clone()
        mask[:, 0] = 0
        mask = mask.unsqueeze(-1)
        hidden = hidden * mask
        return hidden.sum(dim=1) / mask.sum(dim=1).clamp(min=1)

    def get_input_embeddings(self):
        return self.backbone.get_input_embeddings()

    def forward(self, input_ids=None, attention_mask=None, inputs_embeds=None, labels=None, **kwargs):
        outputs = self.backbone(input_ids=input_ids, inputs_embeds=inputs_embeds, attention_mask=attention_mask)
        hidden = outputs["hidden_states"]  # (B, L, H)
        pooled = self.masked_mean_pooling(hidden, attention_mask)  # (B, H)

        logits = self.classifier(pooled)
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)
        return {"loss": loss, "logits": logits}

# ======================
# Load backbone
# ======================
print("Loading fine-tuned classification model directly from Hugging Face ...")

# Directly load the complete fine-tuned model (with classification head) from Hugging Face
config = AutoConfig.from_pretrained(FINETUNED_PATH, trust_remote_code=True)
model = BiMambaForClassification.from_pretrained(
    "eccDNAMamba/eccDNAMamba_1M_cancer_eccdna_prediction_ultra_long",
    config=config,
    trust_remote_code=True,
)

model.eval().to(DEVICE)
print("Model ready.")

# ======================
# Load dataset
# ======================
print("Loading dataset...")
dataset = load_dataset(DATASET_NAME, split="train[:]")  # Only take the first 5 samples as an example
print("Dataset loaded:", len(dataset))

# ======================
# Inference
# ======================
# ===== Dataset wrapper =====
class EccDNADataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        seq, label = self.data[idx]
        aug = circular_augmentation(seq)
        return {
            "input_ids": aug["input_ids"],
            "labels": torch.tensor(label, dtype=torch.long)
        }

# ===== Create dataset =====
data_all = [(sample["sequence"], sample["label"]) for sample in dataset]

_, eval_data_all = train_test_split(
    data_all, test_size=0.2,
    stratify=[l for _, l in data_all],
    random_state=42
)

MIN_LEN, MAX_LEN = 10_000, 200_000
eval_data = [(seq, lab) for seq, lab in eval_data_all if MIN_LEN <= len(seq) <= MAX_LEN]

eval_dataset = EccDNADataset(eval_data[-5:])

# ===== Data collator (automatic padding) =====
data_collator = DataCollatorWithPadding(tokenizer=tokenizer, padding=True, return_tensors="pt")

# ===== Define inference parameters =====
args = TrainingArguments(
    output_dir="./inference_output",
    per_device_eval_batch_size=1,
    dataloader_pin_memory=False,
    bf16=True,
    do_predict=True,
    report_to=[],
)

# ===== Run inference with Trainer =====
trainer = Trainer(
    model=model,
    args=args,
    eval_dataset=eval_dataset,
    data_collator=data_collator,
)

# ===== Parse results =====
output = trainer.predict(eval_dataset)
logits = output.predictions
labels = output.label_ids
probs  = torch.softmax(torch.tensor(logits), dim=1).numpy()
preds  = np.argmax(logits, axis=1)


for i, (label, pred, prob) in enumerate(zip(labels, preds, probs)):
    print(f"[{i}] label={int(label)} pred={int(pred)} probs={prob.tolist()}")

import matplotlib.pyplot as plt

seq_to_explain, true_label = eval_data[1]

# Analysis first 100bp
seq_to_explain = seq_to_explain[:100]
ig_out = integrated_gradients_margin(
    model, tokenizer, seq=seq_to_explain, steps=64,
    baseline_type="pad", reduce="sum",
    ignore_special_tokens=True, target_label=None
)
print(f"pred_label={ig_out['pred_label']}, "
      f"true_label={true_label},"
      f"[margin] logits={ig_out['logits'].tolist()}, "
      f"initial_margin={ig_out['initial_margin']:.4f}")

tokens_vis = ig_out["tokens"]
scores_vis = ig_out["attributions"].numpy()
plot_heatmap_1d(tokens_vis, scores_vis, title="Token-level Attribution Heatmap")


