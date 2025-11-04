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

def plot_heatmap_1d(labels, values, title="Attribution Heatmap"):
    if len(values) == 0:
        print("[warn] Nothing to visualize (possibly all special tokens or empty decoding).")
        return
    max_len = len(labels)
    fig, ax = plt.subplots(figsize=(max_len/6, 2))
    heat = ax.imshow(values[np.newaxis, :], cmap="bwr", aspect="auto", vmin=-1, vmax=1)
    ax.set_xticks(np.arange(max_len))
    ax.set_xticklabels(labels, rotation=90, fontsize=6)
    ax.set_yticks([])
    cbar = plt.colorbar(heat, ax=ax, orientation="vertical")
    cbar.set_label("Integrated Gradients attribution")
    plt.title(title)
    plt.tight_layout()
    plt.show()


# ---- 2) Contrastive IG for margin-based objective (optional) ----
def _scalar_objective_margin(logits: torch.Tensor, target_label: int | None = None):
    """
    For binary classification: margin = logit[target] - logit[1 - target]
    For multi-class: margin = logit[target] - logit[runner_up]
    Returns (scalar, target, other)
    """
    C = logits.size(1)
    if target_label is None:
        target_label = int(torch.argmax(logits, dim=-1).item())
    if C == 2:
        other = 1 - target_label
        return logits[0, target_label] - logits[0, other], target_label, other
    # Multi-class: use runner-up
    vals, idx = torch.sort(logits[0], descending=True)
    top = int(idx[0].item())
    runner = int(idx[1].item())
    other = runner if top == target_label else top
    return logits[0, target_label] - logits[0, other], target_label, other


def integrated_gradients_margin(
    model,
    tokenizer,
    seq: str,
    steps: int = 64,
    baseline_type: str = "pad",
    reduce: str = "sum",
    ignore_special_tokens: bool = True,
    device: torch.device | None = None,
    target_label: int | None = None,
):
    """
    Similar to integrated_gradients, but uses margin (contrastive) as the target scalar.
    Returns the same fields as the original version, plus 'other_label' and 'initial_margin'.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    # Circular augmentation consistent with training
    head = tokenizer(seq, add_special_tokens=True, truncation=False)["input_ids"]
    ids = head + head[1:65]
    input_ids = torch.tensor(ids, dtype=torch.long).unsqueeze(0).to(device)
    attention_mask = torch.ones_like(input_ids)

    embed = model.get_input_embeddings()
    with torch.no_grad():
        input_embeds = embed(input_ids)

    # Baseline
    if baseline_type == "pad" and tokenizer.pad_token_id is not None:
        pad_ids = torch.full_like(input_ids, tokenizer.pad_token_id)
        with torch.no_grad():
            baseline_embeds = embed(pad_ids)
    elif baseline_type in ("pad", "zeros"):
        baseline_embeds = torch.zeros_like(input_embeds)
    else:
        raise ValueError("baseline_type must be 'pad' or 'zeros'")

    # Determine target/other and record initial margin
    with torch.no_grad():
        logits_full = model(inputs_embeds=input_embeds, attention_mask=attention_mask)["logits"]
        scalar0, target_label, other_label = _scalar_objective_margin(logits_full, target_label)

    # Integration
    alphas = torch.linspace(0.0, 1.0, steps+1, device=device)[1:]
    total_grads = torch.zeros_like(input_embeds)
    for a in alphas:
        scaled = baseline_embeds + a * (input_embeds - baseline_embeds)
        scaled.requires_grad_(True)
        out = model(inputs_embeds=scaled, attention_mask=attention_mask)
        logits = out["logits"]
        scalar, _, _ = _scalar_objective_margin(logits, target_label)
        grads = torch.autograd.grad(scalar, scaled, retain_graph=False, create_graph=False)[0]
        total_grads += grads
    avg_grads = total_grads / len(alphas)
    ig = (input_embeds - baseline_embeds) * avg_grads

    # Aggregate to tokens
    if reduce == "sum":
        token_scores = ig.sum(dim=-1).squeeze(0)
    elif reduce == "l2":
        token_scores = torch.linalg.vector_norm(ig.squeeze(0), ord=2, dim=-1)
    else:
        raise ValueError("reduce must be 'sum' or 'l2'")

    ids_vec = input_ids.squeeze(0)
    if ignore_special_tokens:
        specials = {x for x in [tokenizer.cls_token_id, tokenizer.sep_token_id,
                                tokenizer.pad_token_id, tokenizer.bos_token_id,
                                tokenizer.eos_token_id] if x is not None}
        keep = torch.ones_like(token_scores, dtype=torch.bool)
        for i, tok_id in enumerate(ids_vec.tolist()):
            if tok_id in specials:
                keep[i] = False
        token_scores = torch.where(keep, token_scores, torch.zeros_like(token_scores))

    # Normalize
    denom = token_scores.abs().max().clamp(min=1e-8)
    norm_scores = (token_scores / denom).detach().cpu()
    tokens = tokenizer.convert_ids_to_tokens(ids_vec.tolist())

    return {
        "tokens": tokens,
        "input_ids": ids_vec.detach().cpu(),
        "attributions": norm_scores,
        "logits": logits_full.detach().cpu().squeeze(0),
        "pred_label": int(torch.argmax(logits_full, dim=-1).item()),
        "other_label": other_label,
        "initial_margin": float(scalar0.detach().cpu().item())
    }
