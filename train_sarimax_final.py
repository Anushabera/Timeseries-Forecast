import pandas as pd
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima_model import ARMA
from statsmodels.tsa.stattools import adfuller
import pickle

def parser(s):
    return datetime.strptime(s, '%y-%m-%d')

series = pd.read_csv(r'path_of_file', parse_dates=['InvoiceDate'], index_col='InvoiceDate')

series = series.groupby(series.index).sum()
series = series.asfreq(pd.infer_freq(series.index))
series = series.loc["2018-01-01":]
# print(series.head())  # Display first few rows

series_train = series.loc['2018-01-01':series.index.max()]

monthly_series = series_train.resample('M').sum()

monthly_series.to_csv("monthly_data.csv")

print("Monthly data saved successfully from training pipeline.")


# Verify the result
# print(monthly_series.head())
# print(monthly_series.tail())

from statsmodels.tsa.statespace.sarimax import SARIMAX

model = SARIMAX(monthly_series,
                order=(10, 0, 0),                # (p, d, q) values
                seasonal_order=(1, 0, 0, 12),    # (P, D, Q, S) values for yearly seasonality (12 months)
                enforce_stationarity=False,      # Whether to enforce stationarity
                enforce_invertibility=False)     # Whether to enforce invertibility

# Fit the model
result = model.fit()

# Print the model summary
# print(result.summary())

# Save the trained model
with open(r"path\models\sarimax_model.pkl", "wb") as f:
    pickle.dump(result, f)

print("Model saved")