from pathlib import Path
import sys
import pandas as pd


RAW_FILE = Path("data/raw/libreview.csv")

VERY_LOW_THRESHOLD = 54
LOW_THRESHOLD = 70
HIGH_THRESHOLD = 180
VERY_HIGH_THRESHOLD = 250


def parse_number(value):
    if pd.isna(value):
        return None

    value = str(value).strip()

    if value == "":
        return None

    return float(value.replace(",", "."))


def load_libreview_glucose_readings(file_path: Path) -> pd.DataFrame:
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

    # In GlucoPilot we merge historical and scanned glucose into one glucose value.
    df["glucose_mg_dl"] = df["historic_glucose"].combine_first(df["scan_glucose"])

    glucose = df[df["glucose_mg_dl"].notna()].copy()
    glucose = glucose[["timestamp", "glucose_mg_dl"]]
    glucose = glucose.dropna(subset=["timestamp"])
    glucose = glucose.sort_values("timestamp")

    return glucose


def calculate_daily_summary(glucose: pd.DataFrame, date: str) -> dict:
    target_date = pd.to_datetime(date).date()

    day_data = glucose[glucose["timestamp"].dt.date == target_date].copy()

    if day_data.empty:
        raise ValueError(f"No glucose readings found for date {date}")

    values = day_data["glucose_mg_dl"]

    total_readings = len(day_data)

    very_low = (values < VERY_LOW_THRESHOLD).sum()
    low = ((values >= VERY_LOW_THRESHOLD) & (values < LOW_THRESHOLD)).sum()
    in_range = ((values >= LOW_THRESHOLD) & (values <= HIGH_THRESHOLD)).sum()
    high = ((values > HIGH_THRESHOLD) & (values <= VERY_HIGH_THRESHOLD)).sum()
    very_high = (values > VERY_HIGH_THRESHOLD).sum()

    return {
        "date": str(target_date),
        "readings": total_readings,
        "average_glucose": values.mean(),
        "min_glucose": values.min(),
        "max_glucose": values.max(),
        "std_glucose": values.std(),
        "very_low_readings": very_low,
        "low_readings": low,
        "in_range_readings": in_range,
        "high_readings": high,
        "very_high_readings": very_high,
        "very_low_percent": very_low / total_readings * 100,
        "low_percent": low / total_readings * 100,
        "in_range_percent": in_range / total_readings * 100,
        "high_percent": high / total_readings * 100,
        "very_high_percent": very_high / total_readings * 100,
    }


def print_summary(summary: dict):
    print("\nDaily glucose summary")
    print("=====================")
    print(f"Date: {summary['date']}")
    print(f"Readings: {summary['readings']}")

    print("\nGlucose values")
    print("--------------")
    print(f"Average: {summary['average_glucose']:.2f} mg/dL")
    print(f"Minimum: {summary['min_glucose']:.0f} mg/dL")
    print(f"Maximum: {summary['max_glucose']:.0f} mg/dL")
    print(f"Standard deviation: {summary['std_glucose']:.2f} mg/dL")

    print("\nRanges")
    print("------")
    print(f"Very low (<54): {summary['very_low_percent']:.2f}% ({summary['very_low_readings']} readings)")
    print(f"Low (54-69): {summary['low_percent']:.2f}% ({summary['low_readings']} readings)")
    print(f"In range (70-180): {summary['in_range_percent']:.2f}% ({summary['in_range_readings']} readings)")
    print(f"High (181-250): {summary['high_percent']:.2f}% ({summary['high_readings']} readings)")
    print(f"Very high (>250): {summary['very_high_percent']:.2f}% ({summary['very_high_readings']} readings)")


def main():
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python analysis/daily_summary.py YYYY-MM-DD")
        print("\nExample:")
        print("  python analysis/daily_summary.py 2026-06-14")
        return

    date = sys.argv[1]

    glucose = load_libreview_glucose_readings(RAW_FILE)
    summary = calculate_daily_summary(glucose, date)

    print_summary(summary)


if __name__ == "__main__":
    main()