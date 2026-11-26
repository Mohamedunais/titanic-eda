from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw" / "tested.csv"
OUT = ROOT / "data" / "processed" / "titanic_clean.csv"


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.drop_duplicates()

    # Features from text columns
    df["Title"] = df["Name"].str.extract(r",\s*([^\.]+)\.", expand=False).str.strip()
    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
    df["TicketGroupSize"] = df.groupby("Ticket")["Ticket"].transform("count")
    df["HasCabin"] = df["Cabin"].notna().astype(int)
    df["Deck"] = df["Cabin"].str[0].fillna("U")

    # Missing values (flag first, then fill)
    df["AgeMissing"] = df["Age"].isna().astype(int)
    df["Age"] = df["Age"].fillna(df.groupby(["Title", "Pclass"])["Age"].transform("median"))
    df["Age"] = df["Age"].fillna(df["Age"].median())

    df["Fare"] = df["Fare"].fillna(df.groupby("Pclass")["Fare"].transform("median"))
    df["Fare"] = df["Fare"].fillna(df["Fare"].median())

    # Derived numeric features
    df["FareLog"] = np.log1p(df["Fare"])
    df["FarePerPerson"] = df["Fare"] / df["TicketGroupSize"]

    return df


if __name__ == "__main__":
    OUT.parent.mkdir(parents=True, exist_ok=True)
    cleaned = clean(pd.read_csv(RAW))
    cleaned.to_csv(OUT, index=False)
    print(f"Saved {len(cleaned)} rows to {OUT}")