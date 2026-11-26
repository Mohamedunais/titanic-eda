\# Titanic Data Cleaning \& EDA



Cleans the Titanic passenger dataset and explores what relates to survival.



\## Dataset

`tested.csv` (418 passengers, 12 columns), stored in `data/raw/`.



Note: in this file, every woman survived and no man did, which does not match

the historical record. It is fine for practising cleaning and EDA, but it should

not be used to train models.



\## Cleaning decisions

\- Age (21% missing): flagged with `AgeMissing`, then filled with the median by Title x Pclass

\- Fare (1 missing): filled with the median by Pclass

\- Cabin (78% missing): converted to `HasCabin` and `Deck`

\- New features: Title, FamilySize, TicketGroupSize, FareLog, FarePerPerson

\- Fare outliers were kept because they are genuine first-class fares



\## How to run

Requires \[uv](https://docs.astral.sh/uv/).



```

uv run python src/clean.py

```



This writes `data/processed/titanic\_clean.csv`.



To open the EDA notebook:



```

uv run jupyter lab

```



then open `notebooks/Titanic\_Data\_Analysis.ipynb`.



\## Key findings

\- Overall survival in this file is 36.4%

\- Sex perfectly separates survival, which is an artefact of the file

\- Class, port and family size effects mostly reflect the share of women in each group

\- `HasCabin` works as a wealth indicator

