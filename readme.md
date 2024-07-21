# Stock Price Prediction

This project is focused on predicting the next day's stock closing price using historical stock data and various technical indicators. The model is trained using the XGBoost regression algorithm with features including moving averages, RSI, MACD, and daily return.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Model Training](#model-training)
- [Web Application](#web-application)
- [API Endpoints](#api-endpoints)
- [Contributing](#contributing)
- [License](#license)

## Installation

1. Clone the repository:

```bash
git clone https://github.com/yourusername/StockPrediction.git
cd StockPrediction
```
## Create a virtual environment and activate it
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
```
## Install the required packages:
```bash
pip install -r requirements.txt
#Model Training
python train_stock_model.py
#Web Application
python app.py
```
## Navigate to http://127.0.0.1:5000/ in your web browser to access the web interface for predicting stock prices.

## API Endpoints
## Predict Stock Price
## URL: /predict
## Method: POST
## Request:
```bash
{
    "open": 101.0,
    "high": 105.0,
    "low": 99.0,
    "volume": 1234567,
    "close_adjusted": 102.0,
    "moving_avg_10": 100.5,
    "moving_avg_50": 98.3,
    "rsi": 55.0,
    "macd": 0.5,
    "daily_return": 0.01
}

#Response
{
    "prediction": 103.5
}

```