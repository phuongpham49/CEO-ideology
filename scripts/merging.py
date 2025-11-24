import pandas as pd

# -------------------------------------------------------
# 1. Load datasets
# -------------------------------------------------------

# Replace these filenames with your actual file paths
interviews_file = "fake_firm_interviews.csv"          # interview text
pac_file = "synthetic_pac_contributions.csv"          # synthetic PAC data

df_interviews = pd.read_csv(interviews_file)
df_pac = pd.read_csv(pac_file)

print("Interviews:", df_interviews.shape)
print("PAC:", df_pac.shape)


# -------------------------------------------------------
# 2. Compute firm-level ideology score (average score)
# -------------------------------------------------------

firm_scores = (
    df_pac.groupby("firm")["ideology_score"]
    .mean()
    .reset_index()
)

firm_scores.rename(columns={"firm": "company"}, inplace=True)

print("\nFirm ideology scores (first 5):")
print(firm_scores.head())


# -------------------------------------------------------
# 3. Merge interview text with ideology score
# -------------------------------------------------------

df_merged = df_interviews.merge(
    firm_scores,
    on="company",
    how="inner"      # keep only firms appearing in both datasets
)

print("\nMerged dataset:", df_merged.shape)


# -------------------------------------------------------
# 4. Keep only relevant columns for RoBERTa training
# -------------------------------------------------------

df_final = df_merged[["company", "text", "ideology_score"]]

print("\nFinal training dataframe preview:")
print(df_final.head())


# -------------------------------------------------------
# 5. Save final dataset
# -------------------------------------------------------

output_file = "training_data.csv"
df_final.to_csv(output_file, index=False, encoding="utf-8")

print(f"\nSaved training dataset as: {output_file}")
