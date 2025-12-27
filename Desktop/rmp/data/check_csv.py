import pandas as pd
import numpy as np
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path


def round_half_up_scalar(x, ndigits=2):
    q = Decimal("1").scaleb(-ndigits)  # ndigits=2 -> Decimal('0.01')
    return float(Decimal(str(x)).quantize(q, rounding=ROUND_HALF_UP))

# Vectorized for a pandas Series
def round_half_up_series(s: pd.Series, ndigits=2) -> pd.Series:
    return s.map(lambda v: round_half_up_scalar(v, ndigits))

def check(path):
    df = pd.read_csv(path, dtype=str)
    cols = ["A", "B", "C", "Total", "Succ.", "IPP", "D", "F", "INP", "Compl"]
    for c in cols:
        df[c] = pd.to_numeric(df[c])

    # 3) (A+B+C)/Total, rounded to hundredths
    succ_ratio = (df["A"] + df["B"] + df["C"] + df["IPP"]) / df["Total"]
    comp_ratio = (df["A"] + df["B"] + df["C"] + df["D"]+ df["F"]+ df["IPP"] + df["INP"]) / df["Total"]

    df["succ_calc"] = round_half_up_series(succ_ratio, 2)
    df["comp_calc"] = round_half_up_series(comp_ratio, 2)

    df["succ_matches"] = df["succ_calc"].eq(df["Succ."])
    df["comp_matches"] = df["comp_calc"].eq(df["Compl"])
    df["matches"] = df["succ_matches"] & df["comp_matches"]



    # Rows that fail (including cases with missing/zero Total)
    bad = df[~df["matches"]].copy()

    print(f"Total rows: {len(df)}")
    print(f"Matching rows: {df['matches'].sum()}")

    if len(df) == df['matches'].sum():
        print("ALL GOOD!")
    else:
        print("-----NO!!-----")


# 1) read the CSV
folder = Path("data/csv")

for path in folder.iterdir():
    check(path)    
