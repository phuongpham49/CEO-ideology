"""
text_cleaning.py

Utility script for cleaning interview text data for RoBERTa training.
Removes boilerplate, normalizes text, strips noise, and exports
a cleaned CSV file.

Author: Phuong Pham
"""

import re
import pandas as pd


# --------------------------------------------------------
# Cleaning Function
# --------------------------------------------------------
def clean_text(text: str) -> str:
    """Clean interview text for NLP modeling.
    
    Steps:
    - lower-case everything
    - remove URLs, emails, boilerplate
    - remove long numeric noise (ProQuest codes, etc.)
    - remove control characters
    - remove non-language symbols
    - normalize whitespace
    
    Args:
        text (str): Raw input text
        
    Returns:
        str: Cleaned text
    """
    if not isinstance(text, str):
        return ""

    # Lowercase
    text = text.lower()

    # Remove URLs and emails
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"\S+@\S+", " ", text)

    # Remove known boilerplate patterns
    boilerplate_patterns = [
        r"copyright.*?\d{4}",
        r"all rights reserved.*?\.",
        r"retrieved from.*?(?=\s)",
        r"accessed on.*?(?=\s)",
        r"news articles.*?(?=\d{4})",
        r"general interest periodicals.*?(?=\s)",
    ]
    for p in boilerplate_patterns:
        text = re.sub(p, " ", text)

    # Remove 5+ digit numeric codes
    text = re.sub(r"\b\d{5,}\b", " ", text)

    # Remove ASCII control characters
    text = re.sub(r"[\x00-\x1f\x7f-\x9f]", " ", text)

    # Remove non-language symbols
    text = re.sub(r"[^a-z0-9.,!?;:'\"\s-]", " ", text)

    # Normalize multiple spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text


# --------------------------------------------------------
# Main cleaning function
# --------------------------------------------------------
def clean_file(input_path: str, output_path: str):
    """Read input CSV, apply cleaning, and export cleaned version."""

    print(f"Loading data: {input_path}")
    df = pd.read_csv(input_path)

    if "text" not in df.columns:
        raise ValueError("ERROR: Input file must contain a 'text' column.")

    print("Cleaning 'text' column...")
    df["clean_text"] = df["text"].apply(clean_text)

    # Remove empty & too-short rows
    df = df[df["clean_text"].str.len() > 30]

    # Drop duplicate texts
    df = df.drop_duplicates(subset=["clean_text"])

    # Keep only necessary columns
    keep_cols = [c for c in df.columns if c in ["filename", "company", "industry", "clean_text"]]
    df_clean = df[keep_cols].copy()

    print(f"Saving cleaned file to: {output_path}")
    df_clean.to_csv(output_path, index=False, encoding="utf-8")

    print(f"Done! {len(df_clean)} rows saved.")


# --------------------------------------------------------
# Script entry point
# --------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Clean interview text data.")
    parser.add_argument("--input", type=str, required=True, help="Path to raw input CSV.")
    parser.add_argument("--output", type=str, required=True, help="Where to save cleaned CSV.")

    args = parser.parse_args()

    clean_file(args.input, args.output)
