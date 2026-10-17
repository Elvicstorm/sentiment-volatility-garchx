# Sentiment-Aware Realized Volatility Forecasting

Master's thesis code (MSc Data Science, UPC – FIB, October 2025).
Supervisors: Jose Antonio Sánchez Espigares and Ariel Duarte López (Dept. of Statistics and Operations Research).

Does news sentiment help forecast stock volatility? This project extends the classic GARCH(1,1) model with exogenous news signals (**GARCH-X**) and tests hundreds of specifications on four large US companies from different sectors.

## What it does

- **Data integration**: aligns 13 years (2012–2024) of daily prices (Yahoo Finance), macroeconomic series from FRED (incl. the Economic Policy Uncertainty index) and daily news sentiment/attention scores for Apple, ExxonMobil, JPMorgan and Pfizer.
- **Feature engineering**: sign-separated polarity (positive / negative) from FinBERT and RoBERTa scores, EMA smoothing (3, 5 days), seven document-aggregation schemes, news-volume (attention) features.
- **Custom GARCH-X estimation**: Gaussian maximum likelihood written with JAX + SciPy (`minimize` with bounds), robust (sandwich) standard errors, t-stats, p-values and 95% Wald intervals.
- **Model families**: `garch`, `garch+sentiment`, `garch+attention`, `garch+macro` and their combinations.
- **Validation**: time-series cross-validation (`TimeSeriesSplit`), AIC ranking, MAE / MSE / RMSE / MAPE, and residual diagnostics (Ljung-Box, ARCH-LM, Jarque-Bera, ACF, QQ plots).

## Key findings

- Positive news sentiment is a highly significant driver of conditional volatility (e.g. Apple: p = 2.8×10⁻⁶ for positive vs 0.17 for negative polarity), contrary to the usual "bad news matters more" result.
- News volume (attention) adds little once sentiment is included; the EPU macro index correlates with realized volatility but adds limited predictive power.
- RoBERTa polarity with short EMAs and the `('unique', 'WeightedMean')` aggregation is the most robust default across companies.
- Residuals show fat tails: Gaussian GARCH-X still underestimates extreme moves.

## Repository structure

```
src/garchx_models.py            # data loading, feature engineering, GARCH-X estimation, CV, diagnostics
notebooks/Best_models_selection.ipynb   # ranks result tables by AIC (top 5 per company)
notebooks/Plots_best_models.ipynb       # realized vs fitted volatility, residuals, ACF, QQ plots
data/                           # input CSVs (not included, see below)
```

## Usage

```bash
pip install -r requirements.txt
```

Place the input CSVs in the working directory, then:

```python
from src.garchx_models import model_computation

# company, EMA spans, analysis mode ("N" none, "Y" stats, "Y_P" stats + plots),
# model family, train/test split date, aggregation group
model_computation("Apple", [3], "N", "garch+sentiment+attention",
                  "2024-01-02", ("unique", "WeightedMean"))
```

Results are written to `result_table_<company>/`, `white_tests_<company>/` and `plots/`.

## Data

Input files are not included in this repository:

- `AAPL_historical_data.csv`, `XOM_historical_data.csv`, `JPM_historical_data.csv`, `PFE_historical_data.csv`: daily OHLCV from Yahoo Finance.
- `macro_data.csv`: daily-aligned FRED series.
- `sentiment_volume_scores.csv`: daily news sentiment and volume per ISIN. This dataset was provided by the thesis supervisors and is not publicly redistributable.

## Stack

Python · pandas · NumPy · JAX · SciPy · statsmodels · scikit-learn · Matplotlib
