# House Price Prediction — Kaggle Advanced Regression

A principled feature engineering and preprocessing pipeline for the [Kaggle House Prices: Advanced Regression Techniques](https://www.kaggle.com/c/house-prices-advanced-regression-techniques) competition.

The focus of this project is not just getting predictions — it's doing the data preparation *correctly*. Every imputation decision is justified by the data's structure, and every encoding strategy is chosen to match the semantic nature of each feature.

---

## Pipeline overview

```
Raw data (train.csv / test.csv)
        ↓
Missing value analysis — MAR vs MCAR classification per column
        ↓
Principled imputation — regression, domain logic, median, or mode (chosen per feature)
        ↓
Feature engineering — new composite features from existing columns
        ↓
Categorical encoding — mixed strategy (OHE / Ordinal / Count / Frequency / Target)
        ↓
Feature scaling & PCA
        ↓
Baseline modelling with RandomizedSearchCV tuning
```

---

## What makes this pipeline principled

### Missing value imputation

Rather than filling all missing values with the mean or dropping rows, each column's missingness mechanism is analysed first:

| Feature | Mechanism | Strategy |
|---|---|---|
| `LotFrontage` | Missing At Random (MAR) | Regression imputation using correlated features (`1stFlrSF`, `LotArea`, `GrLivArea`, `TotalBsmtSF`) |
| `MasVnrType` / `MasVnrArea` | Structurally missing (no veneer) | Fill with `'None'` / median based on `MasVnrArea == 0` |
| `BsmtQual`, `BsmtCond` | Structurally missing (no basement) | Fill with `'None'` where `TotalBsmtSF == 0` |
| High-null columns (`PoolQC`, `Alley`, `Fence`, `MiscFeature`) | >80% missing, low signal | Dropped |

### Categorical encoding strategy

Different categorical features carry different types of information and warrant different encoding strategies:

| Strategy | Features | Rationale |
|---|---|---|
| One-Hot Encoding | `Street`, `CentralAir`, `MSZoning`, etc. | Low cardinality nominals with no ordinal relationship |
| Ordinal Encoding | `ExterQual`, `KitchenQual`, `BsmtQual`, etc. | Explicit quality scales (Po → Fa → TA → Gd → Ex) |
| Count Encoding | `Exterior1st`, `Exterior2nd`, `GarageType` | Frequency carries implicit signal about material popularity |
| Frequency Encoding | `SaleType`, `RoofMatl`, `HouseStyle` | Relative market prevalence matters more than raw count |
| Target Encoding | `Neighborhood`, `MSSubClass` | High-cardinality features with strong relationship to target |

Custom `CountEncoder` and `FrequencyEncoder` classes are implemented from scratch (scikit-learn compatible interface).

---

## Tech stack

| Component | Technology |
|---|---|
| Language | Python 3 |
| Data manipulation | pandas, NumPy |
| ML & encoding | scikit-learn, category_encoders |
| Visualisation | Matplotlib, Seaborn |
| Environment | Google Colab / Jupyter Notebook |

---

## Repository structure

```
House-Price-Prediction/
├── kaggle_11.py          # Main preprocessing and feature engineering pipeline
├── README.md
└── summary.csv           # Descriptive statistics export (generated at runtime)
```

---

## Key results

- Regression-based imputation for `LotFrontage` outperformed mean/median imputation by preserving the spatial relationship between lot size and frontage
- Mixed encoding strategy preserved semantic information lost by naive one-hot encoding of ordinal quality features
- Custom encoder classes enable clean reuse across train/test splits without leakage

---

## Author

**Reda Alhamwi** — Data Scientist & ML Builder, Dubai UAE  
[LinkedIn](https://linkedin.com/in/reda-alhamwi) · [GitHub](https://github.com/RedaAlhamwi) · [Hugging Face](https://huggingface.co/spaces/Redahamwi1/Data_Story)
