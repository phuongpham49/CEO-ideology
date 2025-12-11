import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from captum.attr import IntegratedGradients


# ============================================================
# 1. LOAD MODEL + TOKENIZER
# ============================================================
MODEL_PATH = "/Users/m1/CEO-ideology/data/trained_roberta_ideology"

device = torch.device("cpu")

print("[INFO] Loading tokenizer and model...")
tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
model = RobertaForSequenceClassification.from_pretrained(MODEL_PATH)
model.to(device)
model.eval()
embedding_layer = model.get_input_embeddings()
print("[INFO] Model loaded.\n")


# ============================================================
# 2. CLEAN ROBERTA TOKENS
# ============================================================
def clean_roberta_tokens(tokens):
    """
    Remove <s>, </s>, <pad>, strip Ġ, and merge subwords.
    """
    clean = []
    current = ""

    for tok in tokens:
        if tok in ["<s>", "</s>", "<pad>"]:
            continue

        if tok.startswith("Ġ"):        # new word
            if current:
                clean.append(current)
            current = tok[1:]          # drop Ġ
        else:
            current += tok             # continuation/subword

    if current:
        clean.append(current)

    return clean


# ============================================================
# 3. FORWARD FUNCTION FOR IG
# ============================================================
def forward_func(inputs_embeds, attention_mask, target_class=None):
    outputs = model(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
    logits = outputs.logits  # (1, num_labels) or (1,1)

    # Regression case
    if logits.shape[-1] == 1:
        return logits.squeeze(-1)

    # Classification case
    if target_class is None:
        target_class = torch.argmax(logits, dim=-1)

    return logits[0, target_class]


# ============================================================
# 4. CORE IG + IMAGE SAVING
# ============================================================
def interpret_text(
    text,
    n_steps: int = 60,
    prefix: str = "ig_example",
    target_class=None,
):
    """
    Compute IG, print token attributions, and save:
      - <prefix>_bar.png      (bar chart)
      - <prefix>_heatmap.png  (word-level heatmap strip)
    Returns (words, normalized_attributions).
    """
    print("\n======================================================")
    print("INPUT TEXT:", text)
    print("======================================================\n")

    encoded = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=256,
        padding="max_length"
    )

    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)

    embeds = embedding_layer(input_ids)
    baseline = torch.zeros_like(embeds)

    ig = IntegratedGradients(lambda emb, mask: forward_func(emb, mask, target_class))

    print("[INFO] Computing Integrated Gradients...")
    atts, delta = ig.attribute(
        embeds,
        baselines=baseline,
        additional_forward_args=(attention_mask,),
        n_steps=n_steps,
        return_convergence_delta=True
    )

    # collapse over embedding dimension
    atts = atts.sum(dim=-1).squeeze(0).detach().cpu().numpy()

    tokens = tokenizer.convert_ids_to_tokens(input_ids.squeeze(0))
    valid_len = int(attention_mask.squeeze(0).sum().item())
    tokens = tokens[:valid_len]
    atts = atts[:valid_len]

    words = clean_roberta_tokens(tokens)

    # map first subtoken attribution to each merged word
    idx = 0
    word_atts = []
    for w in words:
        word_atts.append(atts[idx])
        idx += 1
    word_atts = np.array(word_atts)

    # normalize to [-1,1] for visualization
    max_val = max(abs(word_atts.max()), abs(word_atts.min()))
    norm = word_atts / (max_val + 1e-10)

    # ------- PRINT IN TERMINAL -------
    print("--- TOKEN ATTRIBUTIONS (normalized) ---")
    for w, s in zip(words, norm):
        color = "\033[94m" if s >= 0 else "\033[91m"
        reset = "\033[0m"
        print(f"{color}{w}: {s:.3f}{reset}")
    print()

    # ------- SAVE BAR PLOT -------
    fig, ax = plt.subplots(figsize=(min(12, len(words) * 0.6), 4))
    ax.bar(np.arange(len(words)), norm)
    ax.set_xticks(np.arange(len(words)))
    ax.set_xticklabels(words, rotation=90)
    ax.set_title("Integrated Gradients Attributions")
    ax.set_ylabel("Normalized attribution")
    fig.tight_layout()

    bar_path = f"{prefix}_bar.png"
    fig.savefig(bar_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[SAVED] Bar plot -> {os.path.abspath(bar_path)}")

    # ------- SAVE HEATMAP STRIP -------
    fig, ax = plt.subplots(figsize=(min(12, len(words) * 0.6), 2))
    # 2D array with one row
    heat = norm[np.newaxis, :]
    im = ax.imshow(heat, cmap="bwr", aspect="auto", vmin=-1, vmax=1)
    ax.set_xticks(np.arange(len(words)))
    ax.set_xticklabels(words, rotation=90)
    ax.set_yticks([])
    cbar = fig.colorbar(im, ax=ax, orientation="vertical", fraction=0.04, pad=0.02)
    cbar.set_label("Attribution")

    fig.tight_layout()
    heatmap_path = f"{prefix}_heatmap.png"
    fig.savefig(heatmap_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[SAVED] Heatmap -> {os.path.abspath(heatmap_path)}\n")

    return words, norm


# ============================================================
# 5. SIDE-BY-SIDE COMPARISON IMAGE
# ============================================================
def compare_texts(text1, text2, prefix: str = "ig_compare", n_steps: int = 60):
    """
    Produce a single PNG with two heatmaps: one per sentence.
    Saves:
      - <prefix>_compare.png
    """
    words1, norm1 = interpret_text(text1, n_steps=n_steps, prefix=f"{prefix}_1")
    words2, norm2 = interpret_text(text2, n_steps=n_steps, prefix=f"{prefix}_2")

    # joint figure
    max_len = max(len(words1), len(words2))
    fig, axes = plt.subplots(2, 1, figsize=(min(12, max_len * 0.6), 4))

    # Sentence 1
    heat1 = norm1[np.newaxis, :]
    im1 = axes[0].imshow(heat1, cmap="bwr", aspect="auto", vmin=-1, vmax=1)
    axes[0].set_xticks(np.arange(len(words1)))
    axes[0].set_xticklabels(words1, rotation=90)
    axes[0].set_yticks([])
    axes[0].set_title("Sentence 1")

    # Sentence 2
    heat2 = norm2[np.newaxis, :]
    im2 = axes[1].imshow(heat2, cmap="bwr", aspect="auto", vmin=-1, vmax=1)
    axes[1].set_xticks(np.arange(len(words2)))
    axes[1].set_xticklabels(words2, rotation=90)
    axes[1].set_yticks([])
    axes[1].set_title("Sentence 2")

    fig.colorbar(im2, ax=axes, orientation="vertical", fraction=0.02, pad=0.01)

    fig.tight_layout()
    compare_path = f"{prefix}_compare.png"
    fig.savefig(compare_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[SAVED] Comparison heatmap -> {os.path.abspath(compare_path)}\n")


# ============================================================
# 6. EXAMPLE USAGE
# ============================================================
if __name__ == "__main__":
    t1 = "The firm strongly supports progressive policies and Democratic candidates."
    t2 = "The company opposes regulation and backs Republican economic proposals."

    # Single-sentence images
    interpret_text(t1, prefix="ig_sentence1")
    interpret_text(t2, prefix="ig_sentence2")

    # Side-by-side comparison image
    compare_texts(t1, t2, prefix="ig_demo")

