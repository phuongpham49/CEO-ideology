import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from captum.attr import IntegratedGradients
import matplotlib.pyplot as plt
import numpy as np


# --------------------------------------------------------
# 1. LOAD MODEL + TOKENIZER
# --------------------------------------------------------
MODEL_PATH = "trained_roberta"   # your fine-tuned model folder

tokenizer = RobertaTokenizer.from_pretrained(MODEL_PATH)
model = RobertaForSequenceClassification.from_pretrained(MODEL_PATH)
model.eval()


# --------------------------------------------------------
# 2. FUNCTION TO GET PREDICTION & EMBEDDINGS
# --------------------------------------------------------
def forward_func(input_ids, attention_mask):
    """
    Forward function for Captum.
    Returns the regression output (single value).
    """
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits.squeeze(-1)
    return logits


# --------------------------------------------------------
# 3. INTEGRATED GRADIENTS INTERPRETABILITY FUNCTION
# --------------------------------------------------------
def interpret_text(text, n_steps=50, visualize=True):
    """
    Runs Integrated Gradients and returns token attributions.
    """

    # tokenize
    encoded = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=256,
        padding="max_length"
    )

    input_ids = encoded["input_ids"]
    attention_mask = encoded["attention_mask"]

    # Build IG
    ig = IntegratedGradients(forward_func)

    # Compute attributions
    attributions, delta = ig.attribute(
        inputs=input_ids,
        baselines=torch.zeros_like(input_ids),
        additional_forward_args=(attention_mask,),
        return_convergence_delta=True,
        n_steps=n_steps
    )

    attributions = attributions.sum(dim=-1).squeeze().detach().numpy()

    # decode tokens
    tokens = tokenizer.convert_ids_to_tokens(input_ids.squeeze())

    # normalize for visualization
    attributions = attributions / (np.max(np.abs(attributions)) + 1e-8)

    # -----------------------------
    # Visualization (optional)
    # -----------------------------
    if visualize:
        print("\n=== Integrated Gradients Visualization ===\n")
        for tok, attr in zip(tokens, attributions):
            if tok in ["<s>", "</s>", "<pad>"]:
                continue
            color = "red" if attr > 0 else "blue"
            print(f"{tok:15s}  {attr:+.3f} ({color})")

        # heatmap plot
        plt.figure(figsize=(16, 2))
        plt.bar(range(len(attributions)), attributions)
        plt.title("Integrated Gradients Token Attributions")
        plt.show()

    return tokens, attributions, delta


# --------------------------------------------------------
# 4. EXAMPLE USAGE
# --------------------------------------------------------
if __name__ == "__main__":

    sample_text = """
    Our strategy focuses on reducing regulatory burdens and strengthening American competitiveness.
    We believe lower taxes and fewer compliance constraints help us reinvest in innovation.
    """

    tokens, atts, delta = interpret_text(sample_text)

    print("\nConvergence delta:", delta)
