from flask import Flask, render_template, request, send_file, jsonify
import pandas as pd
import calendar
import numpy as np
import pickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
import math
from datetime import datetime
import os
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.metrics import mean_absolute_percentage_error as mape
os.environ['MPLBACKEND'] = 'Agg'

app = Flask(__name__)

DATA_PATH = r"path_of_file"
MODEL_PATH = r"path_of_model"
PLOT_PATH = "static/forecast.png"

DATA_PATH_1 = r"path_of_file_1"
MODEL_PATH_1 = r"path_of_model_1"
PLOT_PATH_1 = "static/forecast_1.png"

def load_model():
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)
    
def load_model_1():
    with open(MODEL_PATH_1, "rb") as f:
        return pickle.load(f)

def load_and_prepare_data():
    df = pd.read_csv(DATA_PATH, parse_dates=['InvoiceDate'], index_col='InvoiceDate')
    df = df.groupby(df.index).sum()
    df = df.asfreq(pd.infer_freq(df.index))
    df = df.loc["2018-01-02":"2024-12-31"]
    monthly_series = df.resample('M').sum()  # Resample to monthly data
    return monthly_series

def load_and_prepare_data_1():
    df = pd.read_csv(DATA_PATH_1, parse_dates=['Date'], index_col='Date')
    df = df.loc["2022-01-01":"2024-12-01"]
    return df

import pandas as pd
import calendar

def generate_forecast(data, model, n_months, future_exog = None):
    forecast = model.get_forecast(steps=n_months, exog=future_exog)
    forecast_mean = forecast.predicted_mean
    forecast_index = pd.date_range(start='2025-01-01', periods=n_months, freq='ME')
    forecast_mean.index = forecast_index
    conf_int = forecast.conf_int()
    conf_int.index = forecast_index
    return forecast_mean, conf_int

def generate_forecast_1(data, model, n_months):
    last_date = data.index[-1]
    
    # Start forecasting from the next month after the last available data point
    start_date = last_date + pd.DateOffset(months=1)

    forecast = model.get_forecast(steps=n_months)
    forecast_mean = forecast.predicted_mean
    forecast_mean.index = pd.date_range(start=start_date, periods=n_months, freq='ME')  # Start from the next month after the last available date
    
    return forecast_mean, forecast.conf_int()

def load_and_aggregate_data():
    df = pd.read_csv(DATA_PATH)

    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])

    df['Year'] = df['InvoiceDate'].dt.year
    df['Month'] = df['InvoiceDate'].dt.month

    df_2025 = df[df['Year'] == 2025]
    df_2025 = df_2025[df_2025['Month'].isin([1, 2, 3])]

    monthly_data = df_2025.groupby(['Month'])['Taxable Amount'].sum().reset_index()

    monthly_data['Predicted'] = ''

    monthly_data['MonthName'] = monthly_data['Month'].apply(lambda x: calendar.month_name[x])

    monthly_data['Actual'] = monthly_data['Taxable Amount'].apply(lambda x: int(x))  # Convert to integer
    
    model = load_model()

    n_months = 3  
    forecast_mean, _ = generate_forecast(df, model, n_months) 

    print(f"Forecast Mean Index: {forecast_mean.index}")

    for i, row in monthly_data.iterrows():
        month_name = row['MonthName']
        # Create the forecast_date as the last day of the month (month-end)
        forecast_date = pd.to_datetime(f"2025-{row['Month']:02d}-01") + pd.offsets.MonthEnd(0)  # Month-end date
        if forecast_date in forecast_mean.index:
            monthly_data.at[i, 'Predicted'] = f"{forecast_mean.loc[forecast_date]:,.0f}" 

    data_m = monthly_data.to_dict(orient='records')

    for row in data_m:
        row['Actual'] = f"{row['Actual']:,.0f}"

    total_actual = sum(int(row['Actual'].replace(',', '')) for row in data_m if row['Actual'])
    total_predicted = sum(
        int(row['Predicted'].replace(',', '')) for row in data_m if row['Predicted']
    )

    total_row = {
        'Month': '',
        'MonthName': 'Total',
        'Actual': f"{total_actual:,}",
        'Predicted': f"{total_predicted:,}",
    }

    data_m.append(total_row)
    print(monthly_data['Actual'].dtype)
    print(monthly_data['Predicted'].dtype)
    print(data_m)

    return data_m

def calculate_total_accuracy(data_m):
    total_row = next((row for row in data_m if row['MonthName'] == 'Total'), None)
    
    if total_row and total_row['Actual'] and total_row['Predicted']:
        actual_total = int(total_row['Actual'].replace(',', ''))
        predicted_total = int(total_row['Predicted'].replace(',', ''))
        
        if actual_total != 0:
            error = abs((actual_total - predicted_total) / actual_total)
            accuracy = 100 - (error * 100)
            return round(accuracy, 2)
    
    return None

def get_table_data(n_months):
    df = load_and_prepare_data()
    model = load_model()
    forecast_mean, _ = generate_forecast(df, model, n_months)

    years = list(range(2018, 2025 + math.ceil(n_months / 12)))
    table_data = []
    yearly_sales = {year: ["-"] * 12 for year in years}
    yearly_totals = {year: 0 for year in years}

    # Fill actual data (2018 to 2024)
    for year in range(2018, 2025):
        for month in range(12):
            month_str = f"{year}-{month+1:02d}"
            if month_str in df.index.strftime('%Y-%m').tolist():
                value = df.loc[month_str].sum().item()
                yearly_sales[year][month] = f"{value:,.0f}"
                yearly_totals[year] += value

    # Fill forecast data (2025 onwards)
    for i, value in enumerate(forecast_mean):
        year = 2025 + (i // 12) 
        month_index = i % 12 
        yearly_sales[year][month_index] = f"{value:,.0f}" 
        yearly_totals[year] += value

    # Calculate year-over-year variation
    variation_columns = {}
    for year in years[:-1]: 
        variation_columns[year + 1] = []  
        next_year = year + 1 
        for month in range(12):
            prev_value = yearly_sales[year][month]
            curr_value = yearly_sales[next_year][month]

            if prev_value != "-" and curr_value != "-":
                prev_value = float(prev_value.replace(',', ''))
                curr_value = float(curr_value.replace(',', ''))
                if prev_value > 0:
                    variation = ((curr_value - prev_value) / prev_value) * 100
                    variation_columns[next_year].append(f"{int(round(variation))}%")
                else:
                    variation_columns[next_year].append("-")
            else:
                variation_columns[next_year].append("-")

    variation_columns[years[0]] = ["-"] * 12  

    for month_id in range(1, 13):
        row = {"Month ID": month_id}
        for year in years:
            row[str(year)] = yearly_sales[year][month_id - 1]
            if year > years[0]:
                row[f"{year} Variation"] = variation_columns[year][month_id - 1]
        table_data.append(row)

    # Calculate year-over-year variation for total sales
    yearly_variation_totals = {}
    for year in years[:-1]:
        next_year = year + 1
        prev_total = yearly_totals[year]
        curr_total = yearly_totals[next_year]

        if prev_total > 0:
            yearly_variation_totals[next_year] = ((curr_total - prev_total) / prev_total) * 100 
        else:
            yearly_variation_totals[next_year] = "-" 

    total_row = {"Month ID": "Total"}
    for year in years:
        total_row[str(year)] = f"{yearly_totals[year]:,.0f}" 
        if year in yearly_variation_totals:
            total_row[f"{year} Variation"] = f"{yearly_variation_totals[year]:.0f}%"
        else:
            total_row[f"{year} Variation"] = "-"

    table_data.append(total_row)

    return table_data, years

def get_table_data_1(n_months):
    df = load_and_prepare_data_1() 
    model = load_model_1()
    forecast_mean, _ = generate_forecast_1(df, model, n_months)

    years = list(range(2022, 2025 + math.ceil(n_months / 12)))
    table_data = []
    yearly_sales = {year: ["-"] * 12 for year in years}
    yearly_totals = {year: 0 for year in years}

    # Fill actual data (2022 to 2024)
    for year in range(2022, 2025):
        for month in range(12):
            month_str = f"{year}-{month+1:02d}"
            if month_str in df.index.strftime('%Y-%m').tolist():
                value = df.loc[month_str].sum().item()
                yearly_sales[year][month] = f"{value:,.0f}"
                yearly_totals[year] += value

    # Fill forecast data (2025 onwards)
    for i, value in enumerate(forecast_mean):
        year = 2025 + (i // 12)
        month_index = i % 12
        yearly_sales[year][month_index] = f"{value:,.0f}"
        yearly_totals[year] += value

    # Calculate year-over-year variation
    variation_columns = {}
    for year in years[:-1]:  
        variation_columns[year + 1] = []
        next_year = year + 1
        for month in range(12):
            prev_value = yearly_sales[year][month]
            curr_value = yearly_sales[next_year][month]

            if prev_value != "-" and curr_value != "-":
                prev_value = float(prev_value.replace(',', ''))
                curr_value = float(curr_value.replace(',', ''))
                if prev_value > 0:
                    variation = ((curr_value - prev_value) / prev_value) * 100
                    variation_columns[next_year].append(f"{int(round(variation))}%")
                else:
                    variation_columns[next_year].append("-")
            else:
                variation_columns[next_year].append("-")

    variation_columns[years[0]] = ["-"] * 12  

    for month_id in range(1, 13):
        row = {"Month ID": month_id}
        for year in years:
            row[str(year)] = yearly_sales[year][month_id - 1]
            if year > years[0]:
                row[f"{year} Variation"] = variation_columns[year][month_id - 1]
        table_data.append(row)

    # Calculate year-over-year variation for total sales
    yearly_variation_totals = {}
    for year in years[:-1]:
        next_year = year + 1
        prev_total = yearly_totals[year]
        curr_total = yearly_totals[next_year]

        if prev_total > 0:
            yearly_variation_totals[next_year] = ((curr_total - prev_total) / prev_total) * 100 
        else:
            yearly_variation_totals[next_year] = "-"  

    total_row = {"Month ID": "Total"}
    for year in years:
        total_row[str(year)] = f"{yearly_totals[year]:,.0f}"
        if year in yearly_variation_totals:
            total_row[f"{year} Variation"] = f"{yearly_variation_totals[year]:.0f}%"
        else:
            total_row[f"{year} Variation"] = "-" 

    table_data.append(total_row)

    return table_data, years

def save_forecast_plot(actual_sales, forecast_mean, conf_int, n_months):
    plt.figure(figsize=(12, 6))
    plt.plot(actual_sales.index, actual_sales, label='Actual Sales (Till 2024)', color='blue', linewidth=2)
    plt.plot(forecast_mean.index, forecast_mean, label=f'Forecasted Sales ({n_months} Months)', color='red', linestyle='dashed', linewidth=2)
    plt.fill_between(forecast_mean.index, conf_int.iloc[:, 0], conf_int.iloc[:, 1], color='pink', alpha=0.3, label='Confidence Interval')
    # plt.title('Actual vs Forecasted Sales of Retail', fontsize=14)
    # plt.xlabel('Year-Month', fontsize=12)
    # plt.ylabel('Taxable Amount (Sales)', fontsize=12)
    plt.legend(fontsize=10)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(PLOT_PATH)
    plt.close()

def save_forecast_plot_1(actual_sales, forecast_mean, conf_int, n_months):
    plt.figure(figsize=(12, 6))
    plt.plot(actual_sales.index, actual_sales, label='Actual Sales (Till 2024)', color='blue', linewidth=2)
    plt.plot(forecast_mean.index, forecast_mean, label=f'Forecasted Sales ({n_months} Months)', color='red', linestyle='dashed', linewidth=2)
    plt.fill_between(forecast_mean.index, conf_int.iloc[:, 0], conf_int.iloc[:, 1], color='pink', alpha=0.3, label='Confidence Interval')
    # plt.title('Actual vs Forecasted Sales of DNP', fontsize=14)
    # plt.xlabel('Year-Month', fontsize=12)
    # plt.ylabel('Revenue', fontsize=12)
    plt.legend(fontsize=10)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(PLOT_PATH_1)
    plt.close()

import matplotlib.pyplot as plt

def save_yoy_growth_plot(table_data, years):
    total_sales = {}

    for row in table_data:
        if row["Month ID"] == "Total":
            for year in years:
                if str(year) in row and row[str(year)] != "-":
                    total_sales[str(year)] = float(row[str(year)].replace(",", ""))

    last_full_year = None
    for i in range(len(years) - 1, -1, -1):
        if all(row[str(years[i])] != "-" for row in table_data[:-1]):
            last_full_year = years[i]
            break

    filtered_years = [year for year in years if year <= last_full_year]

    yoy_growth = {}
    for i in range(1, len(filtered_years)):
        prev_year = str(filtered_years[i - 1])
        curr_year = str(filtered_years[i])

        if total_sales.get(prev_year, 0) > 0:
            yoy_growth[curr_year] = ((total_sales[curr_year] - total_sales[prev_year]) / total_sales[prev_year]) * 100
        else:
            yoy_growth[curr_year] = 0 

    # Prepare data for line chart
    x_labels = list(yoy_growth.keys())
    y_values = [yoy_growth[year] for year in x_labels]

    # Plotting the line chart with data labels
    plt.figure(figsize=(10, 5))
    plt.plot(x_labels, y_values, marker='o', linestyle='-', color='blue', label='YoY Growth (%)')

    # Add data labels
    for x, y in zip(x_labels, y_values):
        plt.text(x, y, f'{y:.1f}%', ha='center', va='bottom' if y >= 0 else 'top')

    plt.axhline(0, color='gray', linestyle='--', linewidth=0.8)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("static/yoy_growth.png")
    plt.close()
    
def save_yoy_growth_plot_1(table_data_1, years_1):
    total_sales_1 = {}
    
    for row in table_data_1:
        if row["Month ID"] == "Total":
            for year in years_1:
                if str(year) in row and row[str(year)] != "-":
                    total_sales_1[str(year)] = float(row[str(year)].replace(",", ""))

    # Identify the last full year (year with all 12 months of data)
    last_full_year_1 = None
    for i in range(len(years_1) - 1, -1, -1): 
        if all(row[str(years_1[i])] != "-" for row in table_data_1[:-1]):
            last_full_year_1 = years_1[i]
            break

    filtered_years_1 = [year for year in years_1 if year <= last_full_year_1]

    # Compute Year-over-Year Growth (%)
    yoy_growth_1 = {}
    for i in range(1, len(filtered_years_1)):
        prev_year_1 = str(filtered_years_1[i - 1])
        curr_year_1 = str(filtered_years_1[i])

        if total_sales_1[prev_year_1] > 0:
            yoy_growth_1[curr_year_1] = ((total_sales_1[curr_year_1] - total_sales_1[prev_year_1]) / total_sales_1[prev_year_1]) * 100
        else:
            yoy_growth_1[curr_year_1] = 0 

    # Plot YoY Growth for the second dataset
    plt.figure(figsize=(12, 6))
    plt.plot(yoy_growth_1.keys(), yoy_growth_1.values(), marker='o', linestyle='-', color='blue', label="YoY Growth (%)")
    plt.axhline(y=0, color='gray', linestyle='dashed', linewidth=1)
    # plt.title('Year-over-Year Growth of DNP Sales')
    # plt.xlabel('Year')
    # plt.ylabel('Growth (%)')
    plt.xticks(rotation=45)
    plt.legend()

    for year, growth in yoy_growth_1.items():
        plt.text(year, growth, f"{growth:.0f}%", ha='right' if growth < 0 else 'left', va='bottom' if growth >= 0 else 'top', fontsize=10, color='black')


    # Save the plot
    plt.tight_layout()
    plt.savefig("static/yoy_growth_1.png")
    plt.close()

def save_monthly_avg_sales_plot():
    
    df = load_and_prepare_data()

    if 'Taxable Amount' in df.columns:
        df = df[['Taxable Amount']]
    else:
        df['Taxable Amount'] = df.sum(axis=1)  

    df['Year'] = df.index.year
    df['Month'] = df.index.month

    monthly_totals = df.groupby(['Year', 'Month'])['Taxable Amount'].sum().reset_index()

    monthly_avg = monthly_totals.groupby('Month')['Taxable Amount'].mean()

    plt.figure(figsize=(12, 6))

    plt.bar(monthly_avg.index, monthly_avg.values, color='skyblue', edgecolor='blue')

    for i, value in enumerate(monthly_avg.values):
        plt.text(i + 1, value, f"{value:,.0f}", ha='center', va='bottom', fontsize=9, color='black')

    plt.xticks(range(1, 13), ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                              'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])

    # plt.title("Average Monthly Sales", fontsize=14)
    # plt.xlabel("Month", fontsize=12)
    # plt.ylabel("Average Sales", fontsize=12)

    plt.tight_layout()
    plt.savefig("static/monthly_avg_sales.png")
    plt.close()

import matplotlib.pyplot as plt

def save_monthly_avg_sales_plot_1():
    df = load_and_prepare_data_1()

    df['Month'] = df.index.month

    monthly_avg_sales = df.groupby('Month').mean()

    monthly_sales_data = [df[df['Month'] == month]['Revenue'].values for month in range(1, 13)]

    plt.figure(figsize=(12, 6))

    plt.boxplot(monthly_sales_data, positions=range(1, 13), patch_artist=True, 
                boxprops=dict(facecolor="skyblue", color="blue"), 
                whiskerprops=dict(color="blue"), 
                medianprops=dict(color="black"))

    plt.xticks(range(1, 13), ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
    # plt.title('Monthly Average Sales Distribution (Box Plot)')
    # plt.ylabel('Revenue')
    plt.tight_layout()
    plt.savefig("static/monthly_avg_sales_1.png")
    plt.close()

import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

def calculate_accuracy():
    data = load_and_prepare_data()

    monthly_sales = data['Taxable Amount'].resample('M').sum()

    train = monthly_sales.loc[:'2023-12-31']
    test = monthly_sales.loc['2024-01-01':'2024-12-31']

    model = SARIMAX(train,
                    order=(10, 0, 0),
                    seasonal_order=(1, 0, 0, 12),
                    enforce_stationarity=False,
                    enforce_invertibility=False)
    model_fit = model.fit(disp=False)

    forecast = model_fit.get_forecast(steps=12)
    forecast_mean = forecast.predicted_mean
    forecast_mean.index = pd.date_range(start='2024-01-01', periods=12, freq='M')

    test.index = pd.date_range(start='2024-01-01', periods=12, freq='M')

    valid = test != 0
    accuracy_per_month = 100 - (abs(forecast_mean[valid] - test[valid]) / test[valid]) * 100
    average_accuracy = accuracy_per_month.mean()

    return {
        "average_accuracy": '{:.2f}%'.format(average_accuracy)
    }, average_accuracy

def calculate_accuracy_1():
    data_1 = load_and_prepare_data_1()
    model_1 = load_model_1()

    data_1.index = pd.to_datetime(data_1.index)

    monthly_sales_1 = data_1['Revenue']

    monthly_sales_2024_1 = monthly_sales_1.loc['2024-01-01':'2024-12-31']

    monthly_sales_2024_1 = monthly_sales_2024_1.astype(float)

    forecast_1 = model_1.get_forecast(steps=12)
    forecast_mean_1 = forecast_1.predicted_mean

    forecast_mean_1.index = pd.date_range(start='2024-01-01', periods=12, freq='M')

    monthly_sales_2024_1.index = forecast_mean_1.index

    # print("Actual Revenue:\n", monthly_sales_2024_1)
    # print("Forecasted Revenue:\n", forecast_mean_1)

    difference_1 = abs(forecast_mean_1 - monthly_sales_2024_1)

    diff_per_1 = (difference_1 / monthly_sales_2024_1.replace(0, np.nan)) * 100

    diff_per_1 = diff_per_1.fillna(0)

    accuracy_per_month_1 = 100 - diff_per_1
    monthly_accuracy_avg_1 = accuracy_per_month_1.mean()

    # print("Accuracy Per Month:\n", accuracy_per_month_1)
    # print("Final Model Accuracy: ", monthly_accuracy_avg_1)

    return '{:.2f}%'.format(monthly_accuracy_avg_1)

def plot_trend_seasonality():
    df = load_and_prepare_data()

    df = df.interpolate().dropna()
    df = df.resample('M').sum()
    
    decomposition = seasonal_decompose(df, model='additive', period=12)

    plt.figure(figsize=(12, 6))

    plt.subplot(2, 1, 1)
    plt.plot(decomposition.trend, label='Trend', color='blue')
    plt.title('Trend of Retail Sales')
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(decomposition.seasonal, label='Seasonality', color='green')
    plt.title('Seasonality of Retail Sales')
    plt.legend()

    plt.tight_layout()
    
    output_path = "static/trend_seasonality.png"
    plt.savefig(output_path)
    plt.close()

    return output_path 

def plot_trend_seasonality_1():
    df = load_and_prepare_data_1()
    
    decomposition = seasonal_decompose(df, model='additive', period=12)

    trend_fill = decomposition.trend.fillna(method='ffill')
    plt.figure(figsize=(12, 6))

    plt.subplot(2, 1, 1)
    plt.plot(trend_fill, label='Trend', color='red')
    plt.title('Trend of Offtake')
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(decomposition.seasonal, label='Seasonality', color='purple')
    plt.title('Seasonality of Offtake')
    plt.legend()

    plt.tight_layout()
    
    output_path = "static/trend_seasonality_1.png"
    plt.savefig(output_path)
    plt.close()

    return output_path

# #######################################   DEALERS   ###########################################

DEALER_CONFIG = {
    "Dealer_1": {
        "data_path": r"path_of_file_1",
        "model_path": r"path_of_model_1"
    }
}

def load_dealer_data(dealer_id):
    data_path = DEALER_CONFIG[dealer_id]["data_path"]
    data = pd.read_csv(data_path, parse_dates=["InvoiceDate"], index_col="InvoiceDate")
    print(data.columns)
    print("Index:", data.index.name)
    return data

def generate_forecast_plot_dealer(dealer_name):
    import pandas as pd
    import matplotlib.pyplot as plt
    import pickle

    config = DEALER_CONFIG[dealer_name]
    data_path = config["data_path"]
    model_path = config["model_path"]

    df = pd.read_csv(data_path, parse_dates=True, index_col='InvoiceDate')
    df.columns = df.columns.str.strip()
    df = df[['Taxable Amount']].copy()
    df = df.resample('M').sum().interpolate().dropna()

    with open(model_path, 'rb') as f:
        model = pickle.load(f)

    forecast = model.forecast(steps=12)
    future_dates = pd.date_range(start=df.index[-1] + pd.offsets.MonthBegin(), periods=12, freq='MS')

    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df['Taxable Amount'], label='Historical')
    plt.plot(future_dates, forecast, label='Forecast', color='orange')
    # plt.title(f'12-Month Forecast – {dealer_name}')
    # plt.xlabel('Date')
    # plt.ylabel('Taxable Amount')
    plt.legend()
    plt.tight_layout()

    forecast_path = f"static/{dealer_name}_forecast.png"
    plt.savefig(forecast_path)
    plt.close()

    return forecast_path

def get_forecasted_data(dealer_name, month):
    model_path = DEALER_CONFIG[dealer_name]["model_path"]
    with open(model_path, 'rb') as model_file:
        model = pickle.load(model_file)
    
    last_date = pd.to_datetime('today')
    forecast_index = pd.date_range(start=last_date, periods=12, freq='M')

    forecast_values = model.get_forecast(steps=12).predicted_mean  # Use the SARIMAX model to predict the next months
    forecasted_value_for_month = forecast_values[month - 1]  # Get the forecast for the given month
    return forecasted_value_for_month

def generate_forecast_table_dealer(dealer_name, years, n_months=12):
    import pandas as pd
    from dateutil.relativedelta import relativedelta
    import pickle

    config = DEALER_CONFIG[dealer_name]
    df = pd.read_csv(config["data_path"], parse_dates=True, index_col="InvoiceDate")
    df.columns = df.columns.str.strip()
    df = df[['Taxable Amount']].copy()
    df = df.resample('M').sum()

    table_data = []
    yearly_sales = {year: ["-"] * 12 for year in years}
    for ts, value in df['Taxable Amount'].items():
        year, month = ts.year, ts.month
        if year in yearly_sales:
            yearly_sales[year][month - 1] = value

    with open(config["model_path"], "rb") as f:
        model = pickle.load(f)
    forecast = model.get_forecast(steps=n_months)
    forecast_mean = forecast.predicted_mean.values

    forecast_start = df.index[-1] + relativedelta(months=1)
    for i, value in enumerate(forecast_mean):
        date = forecast_start + relativedelta(months=i)
        year, month = date.year, date.month
        if year in yearly_sales:
            yearly_sales[year][month - 1] = value

    for month in range(1, 13):
        row = {"Month ID": f"{month:02d}"}
        for i, year in enumerate(years):
            val = yearly_sales[year][month - 1]
            row[str(year)] = f"{val:,.0f}" if isinstance(val, (int, float)) else "-"

            if i > 0:  # Variation from previous year
                prev_val = yearly_sales[years[i - 1]][month - 1]
                if isinstance(prev_val, (int, float)) and isinstance(val, (int, float)) and prev_val != 0:
                    variation = ((val - prev_val) / prev_val) * 100
                    row[f"{year} Variation"] = f"{round(variation)}%"
                else:
                    row[f"{year} Variation"] = "-"
        table_data.append(row)

    return table_data

import matplotlib.pyplot as plt

def save_yoy_growth_plot_dealer(table_data, dealer_name, years):
    import matplotlib.pyplot as plt

    total_sales = {}

    for year in years:
        year_str = str(year)
        total = 0
        for row in table_data:
            if row["Month ID"] != "Total":
                val = row.get(year_str, 0)
                if isinstance(val, (int, float)):
                    total += val
                elif isinstance(val, str) and val.strip() not in ["-", "-"]:
                    try:
                        total += float(val.replace(",", ""))
                    except:
                        pass
        total_sales[year_str] = total

    print("\nTotal Sales by Year:")
    for y, s in total_sales.items():
        print(f"{y}: {s:,.2f}")

    yoy_growth = {}
    for i in range(1, len(years)):
        prev_year = str(years[i - 1])
        curr_year = str(years[i])

        prev_total = total_sales.get(prev_year, 0)
        curr_total = total_sales.get(curr_year, 0)

        if prev_total > 0:
            yoy_growth[curr_year] = ((curr_total - prev_total) / prev_total) * 100
        else:
            yoy_growth[curr_year] = 0

    print("\nYoY Growth by Year:")
    for y, g in yoy_growth.items():
        print(f"{y}: {g:.2f}%")

    labels = list(yoy_growth.keys())
    growth_values = [yoy_growth[year] for year in labels]

    plt.figure(figsize=(10, 6))
    plt.plot(labels, growth_values, marker='o', linestyle='-', color='blue')

    for i, (x, y) in enumerate(zip(labels, growth_values)):
        plt.text(x, y + 0.5, f"{y:.0f}%", ha='center', va='bottom', fontsize=9, color='black')

    plt.axhline(0, color='gray', linestyle='--')
    plt.title("Year-over-Year Growth (%)")
    # plt.ylabel("Growth %")
    # plt.xlabel("Year")
    plt.grid(False)
    plt.tight_layout()

    yoy_plot_path = f"static/{dealer_name}_yoy_plot.png"
    plt.savefig(yoy_plot_path)
    plt.close()
    return yoy_plot_path

def plot_monthly_average_sales_dealer(data_path, dealer_name):
    df = pd.read_csv(data_path, parse_dates=True, index_col='InvoiceDate')
    df['Taxable Amount'] = df['Taxable Amount']
    df.columns = df.columns.str.strip()

    df = df[['Taxable Amount']].copy()

    df = df.interpolate().dropna()

    df['Month'] = df.index.month

    df['Year'] = df.index.year
    monthly_totals = df.groupby(['Year', 'Month'])['Taxable Amount'].sum().reset_index()

    monthly_avg_sales = monthly_totals.groupby('Month')['Taxable Amount'].mean()

    plt.figure(figsize=(12, 6))

    bars=plt.bar(monthly_avg_sales.index, monthly_avg_sales, color='orange')

    # plt.title(f'Average Monthly Sales Over the Years – {dealer_name}')
    # plt.xlabel('Month')
    # plt.ylabel('Average Monthly Sales')

    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    plt.xticks(monthly_avg_sales.index, month_names)
    for i, value in enumerate(monthly_avg_sales.values):
        plt.text(i + 1, value, f"{value:,.0f}", ha='center', va='bottom', fontsize=9, color='black')

    plt.tight_layout()

    output_path = f"static/{dealer_name}_monthly_avg_sales.png"
    plt.savefig(output_path)
    plt.close()

    return output_path

from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_percentage_error

def compute_forecast_accuracy(dealer_name, order=(10,0,0), seasonal_order=(1,0,0,12)):

    df = pd.read_csv(DEALER_CONFIG[dealer_name]["data_path"], parse_dates=["InvoiceDate"])
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    df.set_index("InvoiceDate", inplace=True)
    
    ts = df["Taxable Amount"].resample("M").sum()

    train = ts[ts.index.year <= 2023]
    test = ts[ts.index.year == 2024]

    model = SARIMAX(train, order=order, seasonal_order=seasonal_order, enforce_stationarity=False, enforce_invertibility=False)
    results = model.fit(disp=False)

    forecast = results.forecast(steps=12)
    forecast.index = test.index  

    comparison_df = pd.DataFrame({
        'Forecast': forecast,
        'Actual': test
    })

    comparison_df['% Error'] = ((comparison_df['Actual'] - comparison_df['Forecast']) / comparison_df['Forecast']) * 100

    comparison_df['Absolute % Error'] = comparison_df['% Error'].abs()

    comparison_df['Accuracy (%)'] = 100 - comparison_df['Absolute % Error']

    comparison_df['Forecast (Formatted)'] = comparison_df['Forecast'].apply(lambda x: '{:,.0f}'.format(x))
    comparison_df['Actual (Formatted)'] = comparison_df['Actual'].apply(lambda x: '{:,.0f}'.format(x))

    comparison_df = comparison_df[['Forecast (Formatted)', 'Actual (Formatted)', '% Error', 'Accuracy (%)']]

    print("\n--- Monthly Forecast Accuracy for 2024 ---")
    print(comparison_df)

    total_forecast = forecast.sum()
    total_actual = test.sum()

    total_error = abs(total_actual - total_forecast)

    overall_accuracy = 100 - ((total_error / total_actual) * 100)

    print("\n--- Overall Forecast Accuracy for 2024 ---")
    print(f"Total Forecasted Sales: {total_forecast:,.0f}")
    print(f"Total Actual Sales:     {total_actual:,.0f}")
    print(f"Absolute Error:         {total_error:,.0f}")
    print(f"Overall Accuracy:       {overall_accuracy:.2f}%")

    return comparison_df, overall_accuracy

def plot_trend_seasonality_dealer(data_path, dealer_name):
    df = pd.read_csv(data_path, parse_dates=True, index_col='InvoiceDate')
    df.columns = df.columns.str.strip()

    df = df[['Taxable Amount']].copy()

    df = df.interpolate().dropna()
    df = df.resample('M').sum()

    decomposition = seasonal_decompose(df, model='additive', period=12)

    plt.figure(figsize=(12, 6))

    plt.subplot(2, 1, 1)
    plt.plot(decomposition.trend, label='Trend', color='blue')
    plt.title(f'Trend of Taxable Amount – {dealer_name}')
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(decomposition.seasonal, label='Seasonality', color='green')
    plt.title(f'Seasonality of Taxable Amount – {dealer_name}')
    plt.legend()

    plt.tight_layout()

    output_path = f"static/{dealer_name}_trend_seasonality.png"
    plt.savefig(output_path)
    plt.close()

    return output_path

# #######################################  END OF DEALERS  ###########################################

import plotly.graph_objects as go
import plotly.io as pio
def generate_dealer_donut_chart(csv_path=r'path_of_file_2'):
    df = pd.read_csv(csv_path, usecols=['Taxable Amount', 'InvoiceDate', 'DealerUpdate'])
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    df['Year'] = df['InvoiceDate'].dt.year
    df = df.rename(columns={'Taxable Amount': 'TaxableAmount', 'DealerUpdate': 'Dealer'})

    latest_year = df['Year'].max()
    recent_data = df[df['Year'] == latest_year]

    grouped = recent_data.groupby('Dealer')['TaxableAmount'].sum()
    total = grouped.sum()
    percentages = (grouped / total) * 100

    fig = go.Figure(data=[go.Pie(
        labels=percentages.index,
        values=percentages.values,
        hole=0.4,
        textinfo='label+percent',
        hoverinfo='label+value'
    )])

    fig.update_layout(title=f"Dealer Taxable Amount Distribution - {latest_year}")

    return pio.to_html(fig, full_html=False, include_plotlyjs='cdn')

@app.route("/", methods=["GET", "POST"])
def home():
    n_months = 12

    if request.method == "POST":
        n_months = int(request.form["months"])

    data = load_and_prepare_data()
    model = load_model()
    future_exog = pd.read_csv(r"path_of_exogeneous_variable", parse_dates=['Actual_date']).set_index('Actual_date').asfreq('MS')
    future_exog = future_exog.loc['2025-01-01':'2025-12-01']
    forecast_mean, conf_int = generate_forecast(data, model, n_months, future_exog = future_exog)
    save_forecast_plot(data.resample("ME").sum(), forecast_mean, conf_int, n_months)
    table_data, years = get_table_data(n_months)

    accuracy_data, monthly_accuracy_avg = calculate_accuracy()
    average_accuracy_1 = calculate_accuracy_1()

    data_1 = load_and_prepare_data_1()
    model_1 = load_model_1()
    forecast_mean_1, conf_int_1 = generate_forecast_1(data_1, model_1, n_months)
    save_forecast_plot_1(data_1, forecast_mean_1, conf_int_1, n_months)
    table_data_1, years_1 = get_table_data_1(n_months)

    save_yoy_growth_plot(table_data, years)
    save_yoy_growth_plot_1(table_data_1, years_1)

    save_monthly_avg_sales_plot()
    save_monthly_avg_sales_plot_1()

    trend_seasonality_path = plot_trend_seasonality()
    trend_seasonality_path_1 = plot_trend_seasonality_1()

    data_m = load_and_aggregate_data()

    accuracy = calculate_total_accuracy(data_m)

    df = pd.read_csv(r'path_of_file_3', parse_dates=["InvoiceDate"])
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    latest_year = int(df["InvoiceDate"].dt.year.max())


    return render_template(
        "index.html",
        latest_year=latest_year,
        data_m = data_m,  
        accuracy=accuracy,
        table_data=table_data,
        years=years,
        table_data_1=table_data_1,
        years_1=years_1,
        forecast_data=accuracy_data,
        average_accuracy=accuracy_data["average_accuracy"],
        forecast_data_1=average_accuracy_1,
        average_accuracy_1=average_accuracy_1,
        trend_seasonality=trend_seasonality_path,
        trend_seasonality_1=trend_seasonality_path_1,
        time=datetime.now().timestamp()
    )

@app.route('/forecast.png')
def forecast_image():
    return send_file(PLOT_PATH, mimetype='image/png')

@app.route('/forecast_1.png')
def forecast_image_1():
    return send_file(PLOT_PATH_1, mimetype='image/png')

@app.route('/monthly_avg_sales.png')
def monthly_avg_sales_image():
    return send_file("static/monthly_avg_sales.png", mimetype='image/png')

@app.route('/monthly_avg_sales_1.png')
def monthly_avg_sales_image_1():
    return send_file("static/monthly_avg_sales_1.png", mimetype='image/png')

@app.route('/trend_seasonality.png')
def trend_seasonality_image():
    return send_file("static/trend_seasonality.png", mimetype='image/png')

@app.route('/trend_seasonality_1.png')
def trend_seasonality_image_1():
    return send_file("static/trend_seasonality_1.png", mimetype='image/png')

@app.route('/dealer', methods=["GET"])
def dealer_page():
    dealer_id = request.args.get('dealer', 'Pollutech')  
    
    if dealer_id not in DEALER_CONFIG:
        return f"Dealer '{dealer_id}' not found.", 404

    data_path = DEALER_CONFIG[dealer_id]["data_path"]
    dealer_name = dealer_id  

    df = pd.read_csv(data_path, parse_dates=True, index_col='InvoiceDate')
    df.columns = df.columns.str.strip()  

    years = sorted(df.index.year.unique())

    plot_path = plot_trend_seasonality_dealer(data_path, dealer_name)
    monthly_avg_sales_plot_path = plot_monthly_average_sales_dealer(data_path, dealer_name)
    forecast_path = generate_forecast_plot_dealer(dealer_name)
    table_data = generate_forecast_table_dealer(dealer_name, years)
    print("Table Data:", table_data)

    columns = list(table_data[0].keys())[1:3]

    yoy_growth_plot = save_yoy_growth_plot_dealer(table_data, dealer_name, years)


    comparison_df, overall_accuracy = compute_forecast_accuracy(dealer_name, order=(10, 0, 0), seasonal_order=(1, 0, 0, 12))


    return render_template('dealer.html', 
                           selected_dealer=dealer_name, 
                           plot_image=plot_path,
                           overall_accuracy=overall_accuracy,
                           monthly_avg_sales_plot=monthly_avg_sales_plot_path,
                           forecast_path = forecast_path,
                           forecast_table=table_data,
                           columns = columns,
                           years = years,
                           yoy_growth_plot=yoy_growth_plot)

if __name__ == '__main__':
    app.run(debug=True)
