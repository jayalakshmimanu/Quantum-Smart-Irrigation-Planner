
import pandas as pd
import numpy as np
from pathlib import Path

# Project directories
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "processed_data"
REPORT_DIR = BASE_DIR / "reports"

OUTPUT_DIR.mkdir(exist_ok=True)
REPORT_DIR.mkdir(exist_ok=True)

# Dataset names
FILES = [
    "allocation_instances.csv",
    "crop_reference.csv",
    "efficiency_reference.csv",
    "locations.csv",
    "soil_reference.csv",
    "weather_daily.csv",
    "weekly_crop_water_demand.csv"
]

data = {}

# 1. Load datasets
print("\n========== DATA LOADING ==========")

for file in FILES:
    df = pd.read_csv(DATA_DIR / file)
    data[file] = df

    print(f"\n{file}")
    print("Shape:", df.shape)
    print("Missing values:", df.isnull().sum().sum())
    print("Duplicate rows:", df.duplicated().sum())

# 2. Remove exact duplicate records
print("\n========== DUPLICATE REMOVAL ==========")

for name, df in data.items():
    before = len(df)
    df = df.drop_duplicates().copy()
    after = len(df)

    data[name] = df

    print(name, "Removed:", before - after)

# 3. Convert date columns
print("\n========== DATE CONVERSION ==========")

date_columns = {
    "weather_daily.csv": ["date"],
    "weekly_crop_water_demand.csv": [
        "planting_date", "week_start"
    ],
    "allocation_instances.csv": ["week_start"]
}

for file, columns in date_columns.items():
    for column in columns:
        data[file][column] = pd.to_datetime(
            data[file][column], errors="coerce"
        )

        print(file, column,
              "Invalid dates:", data[file][column].isna().sum())

# 4. Convert numeric columns safely
print("\n========== NUMERIC VALIDATION ==========")

numeric_columns = {
    "weather_daily.csv": [
        "et0_mm", "rain_mm", "tmax_c", "tmin_c"
    ],
    "weekly_crop_water_demand.csv": [
        "kc_mean", "et0_mm", "etc_mm",
        "rain_mm", "eff_rain_mm", "net_irrig_mm"
    ],
    "allocation_instances.csv": [
        "efficiency", "field_area_ha",
        "net_irrig_mm", "demand_m3",
        "priority_weight", "supply_m3"
    ]
}

for file, columns in numeric_columns.items():
    for column in columns:
        data[file][column] = pd.to_numeric(
            data[file][column], errors="coerce"
        )

# 5. Validate data
print("\n========== DATA VALIDATION ==========")

checks = {
    "weather_daily.csv": {
        "et0_mm": lambda x: x >= 0,
        "rain_mm": lambda x: x >= 0
    },
    "weekly_crop_water_demand.csv": {
        "net_irrig_mm": lambda x: x >= 0,
        "etc_mm": lambda x: x >= 0
    },
    "allocation_instances.csv": {
        "demand_m3": lambda x: x >= 0,
        "supply_m3": lambda x: x >= 0,
        "efficiency": lambda x: x.gt(0) & x.le(1),
        "field_area_ha": lambda x: x > 0
    }
}

for file, rules in checks.items():
    for column, condition in rules.items():
        invalid = (~condition(data[file][column])).sum()
        print(f"{file} | {column} | Invalid: {invalid}")

# 6. Check allocation scenario consistency
allocation = data["allocation_instances.csv"]

scenario_check = allocation.groupby("instance_id").agg(
    crop_count=("crop", "nunique"),
    supply_count=("supply_m3", "nunique")
)

print("\n========== SCENARIO VALIDATION ==========")
print("Total scenarios:", len(scenario_check))
print("Scenarios with 4 crops:",
      (scenario_check["crop_count"] == 4).sum())
print("Scenarios with inconsistent supply:",
      (scenario_check["supply_count"] != 1).sum())

# 7. Save processed datasets
print("\n========== SAVING PROCESSED DATA ==========")

for name, df in data.items():
    output_file = OUTPUT_DIR / name
    df.to_csv(output_file, index=False)
    print("Saved:", output_file.name)

# 8. Generate quality report
report = []

for name, df in data.items():
    report.append({
        "Dataset": name,
        "Rows": len(df),
        "Columns": len(df.columns),
        "Missing_Values": df.isnull().sum().sum(),
        "Duplicate_Rows": df.duplicated().sum()
    })

report_df = pd.DataFrame(report)
report_df.to_csv(
    REPORT_DIR / "data_quality_report.csv",
    index=False
)

print("\n========== PREPROCESSING COMPLETED ==========")
print(report_df.to_string(index=False))
print("\nProcessed files saved in processed_data/")
