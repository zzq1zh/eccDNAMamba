# =========================================
# Minimal Finetuning Script
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
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, matthews_corrcoef, roc_auc_score, roc_curve, average_precision_score, auc, precision_recall_curve

# ======================
# Path Configuration
# ======================
DATASET_NAME  = "eccDNAMamba/cancer_eccdna_prediction_short"
BASEMODEL = "eccDNAMamba/eccDNAMamba-1M"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", DEVICE)

# ======================
# Tokenizer + circular augmentation
# ======================
tokenizer = AutoTokenizer.from_pretrained(BASEMODEL, trust_remote_code=True)

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

        for param in self.backbone.parameters():
            param.requires_grad = False

        if freeze_except_last_k > 0:
            for direction in [self.backbone.mamba_forward.backbone, self.backbone.mamba_backward.backbone]:
                layers = direction.layers  # nn.ModuleList
                for layer in layers[-freeze_except_last_k:]:
                    for param in layer.parameters():
                        param.requires_grad = True

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
config = AutoConfig.from_pretrained(BASEMODEL, trust_remote_code=True)
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

train_data, eval_data = train_test_split(
    data_all, test_size=0.2,
    stratify=[l for _, l in data_all],
    random_state=42
)

data_collator = DataCollatorWithPadding(
    tokenizer=tokenizer,
    padding=True,
    return_tensors="pt",
)

train_dataset = EccDNADataset(train_data)
eval_dataset = EccDNADataset(eval_data)

training_args = TrainingArguments(
    output_dir="finetune_weights/cancer_eccdna_prediction_short",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=64,
    num_train_epochs=3,
    learning_rate=8e-4,
    logging_strategy="steps",
    logging_steps=10,
    disable_tqdm=True,
    logging_first_step=True,
    eval_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=2,
    load_best_model_at_end=True,
    bf16=True,
    report_to="none",
    remove_unused_columns=False,
    label_names=["labels"],
    max_grad_norm=1.0,
    metric_for_best_model="f1_macro",
    greater_is_better=True,
    seed=42,
)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    if isinstance(logits, (list, tuple)):
        logits = logits[0]
    preds = np.argmax(logits, axis=1)
    labels = np.array(labels)

    probs = softmax(logits, axis=1)[:, 1]
    # --- Compute ROC ---
    fpr, tpr, _ = roc_curve(labels, probs)
    roc_auc = auc(fpr, tpr)

    # --- Create figure ---
    fig, ax = plt.subplots(figsize=(6, 5))

    # ===== Left: ROC Curve =====
    ax.plot(fpr, tpr, color='blue', label=f'ROC curve (AUC = {roc_auc:.3f})')
    ax.plot([0, 1], [0, 1], 'k--', label='Random guess')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate (Recall)')
    ax.set_title('ROC Curve')
    ax.legend(loc='lower right')
    ax.grid(alpha=0.3)

    # --- Main title + adjust layout ---
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    # ✅ Save as SVG file (high-quality vector image)
    plt.savefig(f"roc_pr_curves.svg", format="svg", dpi=300)

    plt.show()

    return {
        "accuracy": accuracy_score(labels, preds),
        "f1_macro": f1_score(labels, preds, average='macro'),
        "precision_macro": precision_score(labels, preds, average='macro', zero_division=0),
        "recall_macro": recall_score(labels, preds, average='macro', zero_division=0),
        "f1_micro": f1_score(labels, preds, average='micro'),
        "precision_micro": precision_score(labels, preds, average='micro', zero_division=0),
        "recall_micro": recall_score(labels, preds, average='micro', zero_division=0),
        "mcc": matthews_corrcoef(labels, preds),
        "roc_auc": roc_auc,
    }


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)

trainer.train()

# Evaluate and save
metrics = trainer.evaluate()
# Save the model and state to this run directory
trainer.save_model(os.path.join("finetune_weights/cancer_eccdna_prediction_short", "model"))
trainer.save_state()

