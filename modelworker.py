
from datetime import datetime, timedelta
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf
import lightgbm as lgb
import seaborn as sns
from sklearn.metrics import mean_absolute_error
import joblib
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential, load_model
from keras.layers import LSTM, Dense, Dropout
import plotly.graph_objs as go


def getYahooFinance(against_curr):
    # Define currency symbol lookup function inside getYahooFinance()
    def currSymbol(currency_name):
        currency_dict = {'Euro': 'USDEUR=X', 
                     'Pound': 'USDGBP=X', 
                     'Yen': 'USDJPY=X', 
                     'Canadian Dollar': 'USDCAD=X', 
                     'Swiss Franc': 'USDCHF=X', 
                     'Australian Dollar': 'USDAUD=X'}

        if currency_name in currency_dict:
            return currency_dict[currency_name]

    symbol = currSymbol(against_curr)
    tickers = symbol.split()

    # Convert start date to datetime object
    end_date = datetime.now().replace(hour=0,minute=0,second=0,microsecond=0)
    #end_date=datetime(2023,6,21)
    start_date = calcDatetime().replace(hour=0,minute=0,second=0, microsecond=0)
    #start_date = datetime(2002,6,21)

    interval = '1d'
    exchange_rates = []
    for ticker in tickers:
        t = yf.Ticker(ticker)
        exchange_rates.append(t.history(start=start_date, end=end_date, interval=interval))

    # Create a DataFrame of exchange rates
    ex_df = pd.concat(exchange_rates, axis=1)
    df = ex_df.drop(['Volume', 'Dividends', 'Stock Splits','Open','High','Low'], axis=1)
    return df


def calcDatetime():
    now = datetime.now()-timedelta(days=7)
    five_days_ago = now - timedelta(days=7)
    return datetime(five_days_ago.year, five_days_ago.month, five_days_ago.day)

# Test the function
#df = getYahooFinance('Yen')
#df.head()

def modelLoader(data,model_path):
    # load the saved model
    model = load_model(model_path)
    
    # make predictions on new data
    new_data = data # create a DataFrame with new data
    
    scaler = MinMaxScaler(feature_range=(0,1))
    scaler.fit(new_data) # fit the scaler object to the data
    
    scaled_data = scaler.transform(new_data) # transform the data using the fitted scaler
    
    predictions = model.predict(scaled_data)
    
    # inverse transform the predictions using the fitted scaler
    predictions = scaler.inverse_transform(predictions)
    
    return predictions

def getNumberOfDaysAndData(curr_type,numberOfDays):

    data = getYahooFinance(curr_type)
    # assuming your existing data is in the format of a DataFrame called "data"
    last_date = data.index[-1] # get the last date in the existing data
        # create a new DataFrame with the last 5 days and future 3 days
    last_five_days = data.tail(numberOfDays)
    print(last_five_days)
    future_dates = [last_date + timedelta(days=i+1) for i in range(numberOfDays)]# generate the future dates
    print(future_dates)
    future_data = pd.concat([last_five_days, pd.DataFrame(index=future_dates, columns=data.columns)])
    future_data = future_data.dropna() 


    #future_data = pd.concat([last_five_days, pd.DataFrame(index=future_dates, columns=data.columns)])

    # make predictions using your existing model loader function
    model_path =None
    if curr_type=="Yen":
        model_path="yen_prediction.h5"
    elif curr_type =="Pound":
        model_path="pound_prediction.h5"
    elif curr_type=="Swiss Franc":
        model_path="swiss_franc_prediction.h5"
    elif curr_type=="Australian Dollar":
        model_path="canadian_dollar_prediction.h5"
    else :
        model_path="canadian_dollar_prediction.h5"
    print(model_path)
    predictions = modelLoader(future_data,model_path)

    # return the predictions for the future 3 days
    return data, predictions,future_dates


def plotGraph(currVal, daysVal):
    pred = getNumberOfDaysAndData(currVal, daysVal)
    df = pred[0]
    predictions = pred[1]
    timestamps = pred[2]
    timestamp_strings = [str(ts) for ts in timestamps]
    prediction_values = [p[0] for p in predictions]
    date_index = pd.DatetimeIndex(timestamp_strings).tz_convert('UTC').floor('D')
    df_pred = pd.DataFrame(prediction_values, index=date_index, columns=['Close.1'])

    print(df_pred)
    
    # Merge the two dataframes using outer join based on the date index
    df = pd.merge(df, df_pred, how='outer', left_index=True, right_index=True)

    # Fill missing values in Close column with values from Close.1 column
    df["Close"] = df["Close"].fillna(value=df["Close.1"])

    # Drop Close.1 column
    df.drop(columns=["Close.1"], inplace=True)
    fig_dict = {
    'data': [{
        'y': df['Close'],
        'x': df.index,
        'type': 'scatter',
        'name': 'Value (in USD)',
        'marker': {'color': 'green'} # set default marker color to green
    }],
    'layout': {
        'title': 'DSI Project level 2.2 ',
        'plot_bgcolor': 'rgb(230, 230, 230)',
        'showlegend': True
    }

    }   
    if daysVal <= len(df):
        marker_color = ['red' if x == daysVal else 'green' for x in range(len(df))]
        fig_dict['data'][0]['marker']['color'] = marker_color


    return fig_dict
