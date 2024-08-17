import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from .serializers import SignalResultSerializer

class SignalResultsAPIView(APIView):
    def get(self, request, format=None):
        # Tickers list without extra spaces
        tickers = [
            "HAIL", "HOMZ", "IAI", "IAK", "IAT", "IBB", "ICF", "IDGT", "IDU", "IEDI", "IEO",
            "IETC", "IEZ", "IFRA", "IHE", "IHF", "IHI", "ITA", "ITB", "IYC", "IYE", "IYF", "IYG", "IYH",
            "IYJ", "IYK", "IYM", "IYR", "IYT", "IYW", "IYZ", "KBE", "KBWB", "KBWP", "KBWR",
            "KBWY", "KCE", "KIE", "KRE", "LABD", "LABU", "LTL", "MILN", "MLPA", "MLPX", "MORT"
        ]

        start = "2024-01-01"
        end = "2024-07-31"  # current date
        data = yf.download(tickers, start=start, end=end)

        # Simple Moving Average Calculation
        sma_short = data["Adj Close"].rolling(window=50).mean()
        sma_long = data["Adj Close"].rolling(window=100).mean()

        # RSI Calculation for each ticker
        rsi = pd.DataFrame(index=data.index)

        for ticker in tickers:
            delta = data['Adj Close'][ticker].diff(1)
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            avg_gain = gain.rolling(window=14).mean()
            avg_loss = loss.rolling(window=14).mean()
            rs = avg_gain / avg_loss
            rsi[ticker] = 100 - (100 / (1 + rs))

        # Combine indicators into a single DataFrame
        indicators = pd.concat([sma_short, sma_long, rsi], axis=1, keys=['sma_short', 'sma_long', 'rsi'])
        indicators = indicators.dropna(how='any')

        # Flatten the MultiIndex columns
        indicators.columns = indicators.columns.map('_'.join)

        # Generate trading signals based on moving averages
        for ticker in tickers:
            indicators['signal_' + ticker] = np.where(indicators['sma_short_' + ticker] > indicators['sma_long_' + ticker], 1, 0)

        # Prepare the dataset for training the model
        X_columns = []
        for ticker in tickers:
            X_columns.extend(['sma_short_' + ticker, 'sma_long_' + ticker, 'rsi_' + ticker])

        X = indicators[X_columns]
        y = pd.DataFrame({ticker: indicators['signal_' + ticker] for ticker in tickers})

        # Initialize the models
        models = {
            "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
            "LogisticRegression": LogisticRegression(random_state=42, max_iter=1000),
            "SVC": SVC(kernel='linear', random_state=42)
        }

        # Dictionary to store the final results
        results = []

        # Train and evaluate each model
        for ticker in tickers:
            print(f"Processing ticker: {ticker}...")

            # Extract features for the current ticker
            X_ticker = X[['sma_short_' + ticker, 'sma_long_' + ticker, 'rsi_' + ticker]]
            y_ticker = y[ticker]

            # Split the data into training and testing sets
            X_train, X_test, y_train, y_test = train_test_split(X_ticker, y_ticker, test_size=0.3, random_state=42)

            # Store the original signal result, converted to 'BUY' or 'SELL'
            original_signal = 'BUY' if indicators['signal_' + ticker].iloc[-1] == 1 else 'SELL'

            # Dictionary to store predictions for the current ticker
            ticker_results = {
                'ticker': ticker,
                'original_signal': original_signal
            }

            # Train and predict with each model
            for model_name, model in models.items():
                if len(np.unique(y_train)) > 1:
                    # Train the model
                    model.fit(X_train, y_train)

                    # Predict with the model and convert to 'BUY' or 'SELL'
                    model_prediction = model.predict(X_ticker)[-1]
                    model_prediction = 'BUY' if model_prediction == 1 else 'SELL'

                    # Store the model's prediction in the dictionary
                    ticker_results[model_name + '_prediction'] = model_prediction

                    # Optionally: Calculate and print the accuracy of the model
                    y_pred = model.predict(X_test)
                    accuracy = accuracy_score(y_test, y_pred)
                    print(f"Model accuracy for {ticker} with {model_name}: {accuracy}")
                else:
                    # If there's only one class in y_train, predict using that single class
                    single_class_prediction = 'BUY' if y_train.iloc[0] == 1 else 'SELL'
                    ticker_results[model_name + '_prediction'] = single_class_prediction
                    print(f"Only one class in training data for {ticker} with {model_name}. Predicting {single_class_prediction} for all.")

            # Add the results for the current ticker to the list
            results.append(ticker_results)

        # Serialize the results and send the response
        serializer = SignalResultSerializer(results, many=True)
        return Response(serializer.data, status=status.HTTP_200_OK)
    

#################################### Codigo ultimo dia del mes #################################### 

"""
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from .serializers import SignalResultSerializer
import datetime
import pytz

class SignalResultsAPIView(APIView):
    def get(self, request, format=None):
        # Tickers list without extra spaces
        tickers = [
            "HAIL", "HOMZ", "IAI", "IAK", "IAT", "IBB", "ICF", "IDGT", "IDU", "IEDI", "IEO",
            "IETC", "IEZ", "IFRA", "IHE", "IHF", "IHI", "ITA", "ITB", "IYC", "IYE", "IYF", "IYG", "IYH",
            "IYJ", "IYK", "IYM", "IYR", "IYT", "IYW", "IYZ", "KBE", "KBWB", "KBWP", "KBWR",
            "KBWY", "KCE", "KIE", "KRE", "LABD", "LABU", "LTL", "MILN", "MLPA", "MLPX", "MORT"
        ]

        # Set start date
        start = "2024-01-01"

        # Calculate end date dynamically as the last day of the current month
        tz = pytz.timezone('America/Bogota')
        now = datetime.datetime.now(tz)
        last_day_of_month = datetime.datetime(now.year, now.month, 1, tzinfo=tz) + datetime.timedelta(days=32)
        last_day_of_month = last_day_of_month.replace(day=1) - datetime.timedelta(days=1)
        end = last_day_of_month.strftime('%Y-%m-%d')

        # Fetch data from yfinance
        data = yf.download(tickers, start=start, end=end)

        # Simple Moving Average Calculation
        sma_short = data["Adj Close"].rolling(window=50).mean()
        sma_long = data["Adj Close"].rolling(window=100).mean()

        # RSI Calculation for each ticker
        rsi = pd.DataFrame(index=data.index)

        for ticker in tickers:
            delta = data['Adj Close'][ticker].diff(1)
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            avg_gain = gain.rolling(window=14).mean()
            avg_loss = loss.rolling(window=14).mean()
            rs = avg_gain / avg_loss
            rsi[ticker] = 100 - (100 / (1 + rs))

        # Combine indicators into a single DataFrame
        indicators = pd.concat([sma_short, sma_long, rsi], axis=1, keys=['sma_short', 'sma_long', 'rsi'])
        indicators = indicators.dropna(how='any')

        # Flatten the MultiIndex columns
        indicators.columns = indicators.columns.map('_'.join)

        # Generate trading signals based on moving averages
        for ticker in tickers:
            indicators['signal_' + ticker] = np.where(indicators['sma_short_' + ticker] > indicators['sma_long_' + ticker], 1, 0)

        # Prepare the dataset for training the model
        X_columns = []
        for ticker in tickers:
            X_columns.extend(['sma_short_' + ticker, 'sma_long_' + ticker, 'rsi_' + ticker])

        X = indicators[X_columns]
        y = pd.DataFrame({ticker: indicators['signal_' + ticker] for ticker in tickers})

        # Initialize the models
        models = {
            "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
            "LogisticRegression": LogisticRegression(random_state=42, max_iter=1000),
            "SVC": SVC(kernel='linear', random_state=42)
        }

        # Dictionary to store the final results
        results = []

        # Train and evaluate each model
        for ticker in tickers:
            print(f"Processing ticker: {ticker}...")

            # Extract features for the current ticker
            X_ticker = X[['sma_short_' + ticker, 'sma_long_' + ticker, 'rsi_' + ticker]]
            y_ticker = y[ticker]

            # Split the data into training and testing sets
            X_train, X_test, y_train, y_test = train_test_split(X_ticker, y_ticker, test_size=0.3, random_state=42)

            # Store the original signal result, converted to 'BUY' or 'SELL'
            original_signal = 'BUY' if indicators['signal_' + ticker].iloc[-1] == 1 else 'SELL'

            # Dictionary to store predictions for the current ticker
            ticker_results = {
                'ticker': ticker,
                'original_signal': original_signal
            }

            # Train and predict with each model
            for model_name, model in models.items():
                if len(np.unique(y_train)) > 1:
                    # Train the model
                    model.fit(X_train, y_train)

                    # Predict with the model and convert to 'BUY' or 'SELL'
                    model_prediction = model.predict(X_ticker)[-1]
                    model_prediction = 'BUY' if model_prediction == 1 else 'SELL'

                    # Store the model's prediction in the dictionary
                    ticker_results[model_name + '_prediction'] = model_prediction

                    # Optionally: Calculate and print the accuracy of the model
                    y_pred = model.predict(X_test)
                    accuracy = accuracy_score(y_test, y_pred)
                    print(f"Model accuracy for {ticker} with {model_name}: {accuracy}")
                else:
                    # If there's only one class in y_train, predict using that single class
                    single_class_prediction = 'BUY' if y_train.iloc[0] == 1 else 'SELL'
                    ticker_results[model_name + '_prediction'] = single_class_prediction
                    print(f"Only one class in training data for {ticker} with {model_name}. Predicting {single_class_prediction} for all.")

            # Add the results for the current ticker to the list
            results.append(ticker_results)

        # Serialize the results and send the response
        serializer = SignalResultSerializer(results, many=True)
        return Response(serializer.data, status=status.HTTP_200_OK)

"""