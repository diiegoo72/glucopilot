from pathlib import Path
import sys
import pandas as pd


RAW_FILE = Path("data/raw/libreview.csv")


VERY_LOW_THRESHOLD = 54
LOW_THRESHOLD = 70
HIGH_THRESHOLD = 180
VERY_HIGH_THRESHOLD = 250

MAX_GAP_MINUTES = 30


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

    # In GlucoPilot we merge historical and scanned glucose into one value.
    df["glucose_mg_dl"] = df["historic_glucose"].combine_first(df["scan_glucose"])

    df["rapid_insulin_units"] = df["Insulina de acción rápida (unidades)"].apply(parse_number)
    df["basal_insulin_units"] = df["Insulina de acción larga (unidades)"].apply(parse_number)

    return df.dropna(subset=["timestamp"]).sort_values("timestamp")


def get_glucose_readings(df: pd.DataFrame) -> pd.DataFrame:
    glucose = df[df["glucose_mg_dl"].notna()].copy()
    glucose = glucose[["timestamp", "glucose_mg_dl"]]
    return glucose.sort_values("timestamp")


def get_insulin_doses(df: pd.DataFrame) -> pd.DataFrame:
    rapid = df[df["rapid_insulin_units"].notna()].copy()
    rapid = rapid[["timestamp", "rapid_insulin_units"]]
    rapid = rapid.rename(columns={"rapid_insulin_units": "units"})
    rapid["type"] = "RAPID"

    basal = df[df["basal_insulin_units"].notna()].copy()
    basal = basal[["timestamp", "basal_insulin_units"]]
    basal = basal.rename(columns={"basal_insulin_units": "units"})
    basal["type"] = "BASAL"

    insulin = pd.concat([rapid, basal], ignore_index=True)
    return insulin.sort_values("timestamp")


def classify_glucose_range(value: float) -> str:
    if value < VERY_LOW_THRESHOLD:
        return "very_low"
    if value < LOW_THRESHOLD:
        return "low"
    if value <= HIGH_THRESHOLD:
        return "in_range"
    if value <= VERY_HIGH_THRESHOLD:
        return "high"
    return "very_high"


def calculate_time_weighted_ranges(day_glucose: pd.DataFrame) -> dict:
    day_glucose = day_glucose.sort_values("timestamp").copy()

    day_glucose["next_timestamp"] = day_glucose["timestamp"].shift(-1)
    day_glucose["delta_minutes"] = (
        day_glucose["next_timestamp"] - day_glucose["timestamp"]
    ).dt.total_seconds() / 60

    valid_intervals = day_glucose[
        day_glucose["delta_minutes"].notna()
        & (day_glucose["delta_minutes"] > 0)
        & (day_glucose["delta_minutes"] <= MAX_GAP_MINUTES)
    ].copy()

    ignored_intervals = day_glucose[
        day_glucose["delta_minutes"].notna()
        & (day_glucose["delta_minutes"] > MAX_GAP_MINUTES)
    ].copy()

    minutes = {
        "very_low_minutes": 0.0,
        "low_minutes": 0.0,
        "in_range_minutes": 0.0,
        "high_minutes": 0.0,
        "very_high_minutes": 0.0,
    }

    for _, row in valid_intervals.iterrows():
        glucose_range = classify_glucose_range(row["glucose_mg_dl"])
        key = f"{glucose_range}_minutes"
        minutes[key] += row["delta_minutes"]

    covered_minutes = sum(minutes.values())
    ignored_gap_minutes = ignored_intervals["delta_minutes"].sum()

    def percent(value):
        if covered_minutes == 0:
            return 0
        return value / covered_minutes * 100

    return {
        **minutes,
        "covered_minutes": covered_minutes,
        "ignored_gap_minutes": ignored_gap_minutes,
        "very_low_time_percent": percent(minutes["very_low_minutes"]),
        "low_time_percent": percent(minutes["low_minutes"]),
        "in_range_time_percent": percent(minutes["in_range_minutes"]),
        "high_time_percent": percent(minutes["high_minutes"]),
        "very_high_time_percent": percent(minutes["very_high_minutes"]),
    }


def calculate_daily_summary(df: pd.DataFrame, date: str) -> dict:
    target_date = pd.to_datetime(date).date()

    glucose = get_glucose_readings(df)
    insulin = get_insulin_doses(df)

    day_glucose = glucose[glucose["timestamp"].dt.date == target_date].copy()
    day_insulin = insulin[insulin["timestamp"].dt.date == target_date].copy()

    if day_glucose.empty:
        raise ValueError(f"No glucose readings found for date {date}")

    values = day_glucose["glucose_mg_dl"]
    total_readings = len(day_glucose)

    very_low = (values < VERY_LOW_THRESHOLD).sum()
    low = ((values >= VERY_LOW_THRESHOLD) & (values < LOW_THRESHOLD)).sum()
    in_range = ((values >= LOW_THRESHOLD) & (values <= HIGH_THRESHOLD)).sum()
    high = ((values > HIGH_THRESHOLD) & (values <= VERY_HIGH_THRESHOLD)).sum()
    very_high = (values > VERY_HIGH_THRESHOLD).sum()

    time_weighted = calculate_time_weighted_ranges(day_glucose)

    rapid_doses = day_insulin[day_insulin["type"] == "RAPID"]
    basal_doses = day_insulin[day_insulin["type"] == "BASAL"]

    return {
        "date": str(target_date),
        "readings": total_readings,
        "average_glucose": values.mean(),
        "min_glucose": values.min(),
        "max_glucose": values.max(),
        "std_glucose": values.std(),

        # Reading-based ranges
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

        # Time-weighted ranges
        **time_weighted,

        # Insulin
        "rapid_dose_count": len(rapid_doses),
        "rapid_units_total": rapid_doses["units"].sum() if not rapid_doses.empty else 0,
        "basal_dose_count": len(basal_doses),
        "basal_units_total": basal_doses["units"].sum() if not basal_doses.empty else 0,
        "insulin_doses": day_insulin,
    }


def format_minutes(minutes: float) -> str:
    hours = int(minutes // 60)
    mins = int(round(minutes % 60))

    if hours == 0:
        return f"{mins} min"

    return f"{hours} h {mins} min"


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

    print("\nRanges by number of readings")
    print("----------------------------")
    print(f"Very low (<54): {summary['very_low_percent']:.2f}% ({summary['very_low_readings']} readings)")
    print(f"Low (54-69): {summary['low_percent']:.2f}% ({summary['low_readings']} readings)")
    print(f"In range (70-180): {summary['in_range_percent']:.2f}% ({summary['in_range_readings']} readings)")
    print(f"High (181-250): {summary['high_percent']:.2f}% ({summary['high_readings']} readings)")
    print(f"Very high (>250): {summary['very_high_percent']:.2f}% ({summary['very_high_readings']} readings)")

    print("\nTime-weighted ranges")
    print("--------------------")
    print(f"Covered time: {format_minutes(summary['covered_minutes'])}")
    print(f"Ignored gaps: {format_minutes(summary['ignored_gap_minutes'])}")

    print(
        f"Very low (<54): {summary['very_low_time_percent']:.2f}% "
        f"({format_minutes(summary['very_low_minutes'])})"
    )
    print(
        f"Low (54-69): {summary['low_time_percent']:.2f}% "
        f"({format_minutes(summary['low_minutes'])})"
    )
    print(
        f"In range (70-180): {summary['in_range_time_percent']:.2f}% "
        f"({format_minutes(summary['in_range_minutes'])})"
    )
    print(
        f"High (181-250): {summary['high_time_percent']:.2f}% "
        f"({format_minutes(summary['high_minutes'])})"
    )
    print(
        f"Very high (>250): {summary['very_high_time_percent']:.2f}% "
        f"({format_minutes(summary['very_high_minutes'])})"
    )

    print("\nInsulin")
    print("-------")
    print(f"Rapid doses: {summary['rapid_dose_count']}")
    print(f"Rapid total units: {summary['rapid_units_total']:.2f} U")
    print(f"Basal doses: {summary['basal_dose_count']}")
    print(f"Basal total units: {summary['basal_units_total']:.2f} U")

    if not summary["insulin_doses"].empty:
        print("\nInsulin dose details")
        print("--------------------")
        for _, row in summary["insulin_doses"].iterrows():
            time = row["timestamp"].strftime("%H:%M")
            print(f"{time} - {row['type']} - {row['units']:.2f} U")


def main():
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python analysis/daily_summary.py YYYY-MM-DD")
        print("\nExample:")
        print("  python analysis/daily_summary.py 2026-06-20")
        return

    date = sys.argv[1]

    df = load_libreview_data(RAW_FILE)
    summary = calculate_daily_summary(df, date)

    print_summary(summary)


if __name__ == "__main__":
    main()