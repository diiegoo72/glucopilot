from pathlib import Path
import sys
import pandas as pd
import matplotlib.pyplot as plt


RAW_FILE = Path("data/raw/libreview.csv")
OUTPUT_DIR = Path("outputs")


LOW_THRESHOLD = 70
HIGH_THRESHOLD = 180


def parse_number(value):
    if pd.isna(value):
        return None

    value = str(value).strip()

    if value == "":
        return None

    return float(value.replace(",", "."))


def load_libreview_data(file_path: Path) -> pd.DataFrame:
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    df = pd.read_csv(file_path, header=1, dtype=str)

    df["timestamp"] = pd.to_datetime(
        df["Sello de tiempo del dispositivo"],
        format="%d-%m-%Y %H:%M",
        errors="coerce"
    )

    df["historic_glucose"] = df["Historial de glucosa mg/dL"].apply(parse_number)
    df["scan_glucose"] = df["Escaneo de glucosa mg/dL"].apply(parse_number)
    df["glucose_mg_dl"] = df["historic_glucose"].combine_first(df["scan_glucose"])

    df["rapid_insulin_units"] = df["Insulina de acción rápida (unidades)"].apply(parse_number)
    df["basal_insulin_units"] = df["Insulina de acción larga (unidades)"].apply(parse_number)

    return df.dropna(subset=["timestamp"]).sort_values("timestamp")


def get_daily_glucose(df: pd.DataFrame, date: str) -> pd.DataFrame:
    target_date = pd.to_datetime(date).date()

    glucose = df[df["glucose_mg_dl"].notna()].copy()
    glucose = glucose[glucose["timestamp"].dt.date == target_date]
    glucose = glucose[["timestamp", "glucose_mg_dl"]].sort_values("timestamp")

    if glucose.empty:
        raise ValueError(f"No glucose readings found for date {date}")

    return glucose


def get_daily_insulin(df: pd.DataFrame, date: str) -> pd.DataFrame:
    target_date = pd.to_datetime(date).date()

    rapid = df[df["rapid_insulin_units"].notna()].copy()
    rapid = rapid[["timestamp", "rapid_insulin_units"]]
    rapid = rapid.rename(columns={"rapid_insulin_units": "units"})
    rapid["type"] = "RAPID"

    basal = df[df["basal_insulin_units"].notna()].copy()
    basal = basal[["timestamp", "basal_insulin_units"]]
    basal = basal.rename(columns={"basal_insulin_units": "units"})
    basal["type"] = "BASAL"

    insulin = pd.concat([rapid, basal], ignore_index=True)
    insulin = insulin[insulin["timestamp"].dt.date == target_date]
    insulin = insulin.sort_values("timestamp")

    return insulin


def plot_daily_glucose(date: str):
    df = load_libreview_data(RAW_FILE)
    glucose = get_daily_glucose(df, date)
    insulin = get_daily_insulin(df, date)

    OUTPUT_DIR.mkdir(exist_ok=True)

    fig, ax = plt.subplots(figsize=(14, 6))

    glucose_for_plot = insert_gaps(glucose, max_gap_minutes=30)

    ax.plot(
        glucose_for_plot["timestamp"],
        glucose_for_plot["glucose_mg_dl"],
        linewidth=1.5,
        label="Glucose"
    )

    ax.axhline(LOW_THRESHOLD, linestyle="--", linewidth=1, label="Low threshold")
    ax.axhline(HIGH_THRESHOLD, linestyle="--", linewidth=1, label="High threshold")

    for _, row in insulin.iterrows():
        ax.axvline(row["timestamp"], linestyle=":", linewidth=1)

        label = f"{row['type']} {row['units']:.1f}U"
        ax.text(
            row["timestamp"],
            glucose["glucose_mg_dl"].max() + 10,
            label,
            rotation=90,
            verticalalignment="bottom",
            fontsize=8
        )

    ax.set_title(f"Daily glucose curve - {date}")
    ax.set_xlabel("Time")
    ax.set_ylabel("Glucose (mg/dL)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.autofmt_xdate()

    output_path = OUTPUT_DIR / f"daily_glucose_{date}.png"
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)

    print(f"Plot saved to: {output_path}")


def insert_gaps(glucose: pd.DataFrame, max_gap_minutes: int = 30) -> pd.DataFrame:
    glucose = glucose.sort_values("timestamp").copy()

    rows = []
    previous_timestamp = None

    for _, row in glucose.iterrows():
        current_timestamp = row["timestamp"]

        if previous_timestamp is not None:
            gap_minutes = (current_timestamp - previous_timestamp).total_seconds() / 60

            if gap_minutes > max_gap_minutes:
                rows.append({
                    "timestamp": previous_timestamp + pd.Timedelta(minutes=1),
                    "glucose_mg_dl": None
                })

        rows.append({
            "timestamp": current_timestamp,
            "glucose_mg_dl": row["glucose_mg_dl"]
        })

        previous_timestamp = current_timestamp

    return pd.DataFrame(rows)


def main():
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python analysis/plot_daily_glucose.py YYYY-MM-DD")
        print("\nExample:")
        print("  python analysis/plot_daily_glucose.py 2026-06-20")
        return

    date = sys.argv[1]
    plot_daily_glucose(date)


if __name__ == "__main__":
    main()