from pathlib import Path
import pandas as pd


RAW_FILE = Path("data/raw/libreview.csv")


def parse_number(value):
    if pd.isna(value):
        return None
    value = str(value).strip()
    if value == "":
        return None
    return float(value.replace(",", "."))


def main():
    if not RAW_FILE.exists():
        raise FileNotFoundError(f"File not found: {RAW_FILE}")

    df = pd.read_csv(RAW_FILE, header=1, dtype=str)

    print("Rows:", len(df))
    print("Columns:")
    for col in df.columns:
        print("-", col)

    print("\nRecord types:")
    print(df["Tipo de registro"].value_counts(dropna=False))

    df["timestamp"] = pd.to_datetime(
        df["Sello de tiempo del dispositivo"],
        format="%d-%m-%Y %H:%M",
        errors="coerce"
    )

    # We merge historical glucose and scanned glucose into one glucose value.
    df["historic_glucose"] = df["Historial de glucosa mg/dL"].apply(parse_number)
    df["scan_glucose"] = df["Escaneo de glucosa mg/dL"].apply(parse_number)

    df["glucose_mg_dl"] = df["historic_glucose"].combine_first(df["scan_glucose"])

    glucose = df[df["glucose_mg_dl"].notna()].copy()
    glucose = glucose[["timestamp", "glucose_mg_dl"]].sort_values("timestamp")

    print("\nGlucose readings:", len(glucose))
    print("First glucose timestamp:", glucose["timestamp"].min())
    print("Last glucose timestamp:", glucose["timestamp"].max())
    print("Min glucose:", glucose["glucose_mg_dl"].min())
    print("Max glucose:", glucose["glucose_mg_dl"].max())
    print("Average glucose:", round(glucose["glucose_mg_dl"].mean(), 2))

    df["rapid_insulin_units"] = df["Insulina de acción rápida (unidades)"].apply(parse_number)
    df["basal_insulin_units"] = df["Insulina de acción larga (unidades)"].apply(parse_number)

    rapid = df[df["rapid_insulin_units"].notna()][["timestamp", "rapid_insulin_units"]]
    basal = df[df["basal_insulin_units"].notna()][["timestamp", "basal_insulin_units"]]

    print("\nRapid insulin records:", len(rapid))
    print("Basal insulin records:", len(basal))

    # Create a small anonymized sample for GitHub.
    sample = glucose.head(300).copy()
    if not sample.empty:
        first_time = sample["timestamp"].min()
        sample["timestamp"] = sample["timestamp"] - first_time
        sample["timestamp"] = pd.Timestamp("2026-01-01") + sample["timestamp"]

        sample_path = Path("data/sample/glucose_sample.csv")
        sample.to_csv(sample_path, index=False)
        print(f"\nSample file created: {sample_path}")


if __name__ == "__main__":
    main()