# 📈 Timeseries Forecasting for Sales

A dynamic web application that forecasts next-year sales with **93% accuracy**, using advanced time series models and exogenous variables.

## 🔍 Overview

This project provides accurate sales forecasts using state-of-the-art techniques in time series analysis, including SARIMAX and Facebook Prophet. It supports external variables (exogenous factors) to enhance prediction reliability.

## 🧰 Tech Stack

- **Python**
- **Data Science & Machine Learning**
- **SARIMAX**
- **Prophet**
- **Flask** 
- **Plotly / Matplotlib**

## 🚀 Features

- User input for entering no. of months and generate forecast
- Forecast future sales trends with 93% accuracy
- Support for exogenous variables
- Interactive web-based interface
- Dynamic visualization of forecast results
- Selection option for dealers
- Personalized forecasting for dealers

## 📦 Installation

```bash
git clone https://github.com/Anushabera/timeseries-forecast-sales.git
cd timeseries-forecast-sales

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the app
python app.py
