from yieldlite.io.final_dataset import load_final_dataset

df = load_final_dataset("data/processed/dataset_final.csv")
print(df.shape)
print(df["panel_id"].nunique(), "panels")
print(df["code_flag"].value_counts())
print(df["district_mismatch"].sum(), "district mismatches")
print(df["commune_corrupted"].sum(), "corrupted commune names")
print(df["variety_flag"].value_counts())
