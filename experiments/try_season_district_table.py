"""Try the SeasonDistrictTable baseline on real data.

Exploratory only - nothing here is part of the tested package.
"""

from yieldlite.io.final_dataset import load_final_dataset
from season_district_table import SeasonDistrictTable

df = load_final_dataset("data/processed/dataset_final.csv")

model = SeasonDistrictTable(shrinkage=10).fit(df, df["yield_t_ha"])
preds = model.predict(df)

mae = (preds - df["yield_t_ha"]).abs().mean()
print("in-sample MAE:", round(mae, 4))
print("overall mean:", round(model.overall_mean_, 3))
print()
print("a few group means:")
for key, value in list(model.group_means_.items())[:6]:
    print(" ", key, round(value, 3))
