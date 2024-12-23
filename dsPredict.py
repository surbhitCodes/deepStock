import numpy as np
import pandas as pd
from pymongo import MongoClient
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
from news_scraper import NewsScraper  # Ensure this is correctly imported


class StockPredict:
    def __init__(self, stock_name, history_filepath, db_name='deepStockDB', collection_name='news_collection',
                 db_host='localhost', db_port=27017):
        """
        :param stock_name: name of the stock
        :param history_filepath: path of the history file
        :param db_name: database name (mongodb)
        :param collection_name: name of the collection in mongo db
        :param db_host: hostname of database
        :param db_port: port of database
        """
        self.stock_name = stock_name
        self.history_filepath = history_filepath
        self.sentiment_analyzer = SentimentIntensityAnalyzer()
        self.historical_data = pd.read_csv(self.history_filepath)
        self.db = MongoClient(db_host, db_port)[db_name]
        self.news_collection = self.db[collection_name]

        # Initialize scalers
        self.price_scaler = MinMaxScaler(feature_range=(0, 1))
        self.sentiment_scaler = MinMaxScaler(feature_range=(0, 1))

        # Placeholder for model and scalers
        self.model = None
        self.scaler = None

        # Preprocess data
        self.preProcess()
        self.add_features()

    def preProcess(self):
        """
        Preprocess stock price data (sort by date, normalize, etc.)
        """
        # Ensure the date column exists
        if 'Date' not in self.historical_data.columns:
            raise KeyError(f"Date column 'Date' not found in the CSV file. Available columns: {self.historical_data.columns}")

        self.historical_data['Date'] = pd.to_datetime(self.historical_data['Date'])
        self.historical_data = self.historical_data.sort_values(by='Date').reset_index(drop=True)
        
        price_features = ['Open', 'High', 'Low', 'Close', 'Volume']
        # Check if all required price features are present
        missing_features = [feat for feat in price_features if feat not in self.historical_data.columns]
        if missing_features:
            raise KeyError(f"Missing price features in the CSV file: {missing_features}")

        self.historical_data[price_features] = self.price_scaler.fit_transform(
            self.historical_data[price_features]
        )

    def add_features(self):
        """
        Add moving averages and volatility as additional features
        """
        df = self.historical_data.copy()
        df['MA7'] = df['Close'].rolling(window=7).mean()
        df['MA21'] = df['Close'].rolling(window=21).mean()
        df['Volatility'] = df['Close'].rolling(window=7).std()

        # Merge with sentiment scores
        sentiment_scores = []
        for date in tqdm(df['Date'], desc="Calculating sentiment scores"):
            score = self.getSentimentScore(date)
            sentiment_scores.append(score)
        
        df['Sentiment'] = sentiment_scores

        # Fill NaN values resulting from rolling calculations
        df.fillna(0, inplace=True)

        # Scale sentiment
        df['Sentiment'] = self.sentiment_scaler.fit_transform(df[['Sentiment']])

        self.historical_data = df

    def getSentimentScore(self, date):
        """
        Query the news_collection for articles around 'date' that mention self.stock_name.
        Compute an aggregated sentiment score using VADER.
        :param date: date for sentiment analysis (pd.Timestamp)
        :return: Total sentiment score (float)
        """
        # Define date range (+/- 1 day)
        start_date = date - pd.Timedelta(days=1)
        end_date = date + pd.Timedelta(days=1)

        # Query MongoDB for relevant articles
        articles = self.news_collection.find(
            {
                'news_heading': {'$regex': self.stock_name, '$options': 'i'},
                'date': {
                    '$gte': start_date,
                    '$lte': end_date
                }
            }
        )

        sentiment_score = 0.0
        count = 0
        for article in articles:
            text = article.get('news_heading', '')
            if text:
                sentiment = self.sentiment_analyzer.polarity_scores(text)['compound']
                sentiment_score += sentiment
                count += 1

        if count == 0:
            return 0.0  # Neutral sentiment if no articles
        else:
            return sentiment_score / count  # Average sentiment

    def createDataset(self, timesteps=60):
        """
        Create features (X) and targets (y) for LSTM model.
        :param timesteps: Number of past days to consider for each input
        :return: X, y as numpy arrays
        """
        df = self.historical_data.copy()
        feature_columns = ['Open', 'High', 'Low', 'Close', 'Volume', 'MA7', 'MA21', 'Volatility', 'Sentiment']
        data = df[feature_columns].values

        X = []
        y = []
        for i in range(timesteps, len(data)):
            X.append(data[i - timesteps:i])
            y.append(data[i, 3])  # Assuming 'Close' is the target

        X = np.array(X)
        y = np.array(y)

        return X, y

    def trainModel(self, timesteps=60, epochs=20, batch_size=32):
        """
        Train an LSTM model on historical data.
        :param timesteps: Number of past days to consider for each input
        :param epochs: Number of training epochs
        :param batch_size: Size of training batches
        """
        # Create dataset
        X, y = self.createDataset(timesteps=timesteps)

        # Split into train and test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, shuffle=False
        )

        # Define the model
        model = Sequential()
        model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
        model.add(Dropout(0.2))
        model.add(LSTM(units=50, return_sequences=False))
        model.add(Dropout(0.2))
        model.add(Dense(units=1))  # Predicting 'Close' price

        model.compile(optimizer='adam', loss='mean_squared_error')

        # Train the model
        history = model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_test, y_test),
            verbose=1
        )

        # Save the model and scalers
        self.model = model
        self.scaler = self.price_scaler  # If you have additional scalers, handle them accordingly

        # Optionally, plot training history
        plt.figure(figsize=(12, 6))
        plt.plot(history.history['loss'], label='Train Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Model Loss During Training')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()

    def latestData(self, timesteps=60):
        """
        Prepare the latest data for prediction.
        :param timesteps: Number of past days to consider
        :return: numpy array suitable for model input
        """
        df = self.historical_data.copy()
        feature_columns = ['Open', 'High', 'Low', 'Close', 'Volume', 'MA7', 'MA21', 'Volatility', 'Sentiment']
        data = df[feature_columns].values

        if len(data) < timesteps:
            raise ValueError("Not enough data to create input for prediction.")

        latest_features = data[-timesteps:]
        latest_features = np.expand_dims(latest_features, axis=0)  # Shape: (1, timesteps, features)

        return latest_features

    def predict(self, data):
        """
        Make a prediction using the trained model.
        :param data: numpy array of shape (1, timesteps, features)
        :return: Predicted 'Close' price (scaled)
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet.")

        predictions = self.model.predict(data)
        return predictions[0][0]  # Return scalar value

    def inverse_scale_prediction(self, scaled_pred):
        """
        Inverse transform the scaled prediction to original scale.
        Handles both single values and arrays.
        :param scaled_pred: Scaled prediction value or array
        :return: Original scale prediction
        """
        # 'Close' is the 4th feature, index 3
        # Since MinMaxScaler was fit on all price features, we need to inverse transform accordingly
        if np.isscalar(scaled_pred):
            dummy = np.zeros((1, 5))  # 5 price features: ['Open', 'High', 'Low', 'Close', 'Volume']
            dummy[0, 3] = scaled_pred  # Set 'Close' value
            original = self.price_scaler.inverse_transform(dummy)
            return original[0, 3]
        else:
            # Assume scaled_pred is an array
            scaled_pred = np.array(scaled_pred).reshape(-1, 1)
            dummy = np.zeros((len(scaled_pred), 5))
            dummy[:, 3] = scaled_pred[:, 0]
            original = self.price_scaler.inverse_transform(dummy)
            return original[:, 3]

    def plot_predictions(self, y_true, y_pred, title='Stock Price Prediction'):
        """
        Plot true vs predicted prices.
        :param y_true: Actual 'Close' prices
        :param y_pred: Predicted 'Close' prices
        :param title: Plot title
        """
        plt.figure(figsize=(12,6))
        plt.plot(y_true, label='Actual Close Prices')
        plt.plot(y_pred, label='Predicted Close Prices')
        plt.title(title)
        plt.xlabel('Time')
        plt.ylabel('Price')
        plt.legend()
        plt.show()


if __name__ == '__main__':
    # Update DB with the latest news
    NewsScraper().update_news()  
    
    # Instantiate predictor
    predictor = StockPredict('sbi', 'data/sbi.csv')
    
    # Train model
    predictor.trainModel()
    
    # Make prediction on the latest data
    latest_features = predictor.latestData()
    scaled_prediction = predictor.predict(latest_features)
    predicted_close = predictor.inverse_scale_prediction(scaled_prediction)
    
    print(f'Predicted Close Price for the next day: {predicted_close}')
    
    # Evaluation on Test Data and Plotting
    # Recreate the dataset
    X, y = predictor.createDataset()
    
    # Split into train and test (same as in trainModel)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )
    
    # Make predictions on the test set
    test_predictions_scaled = predictor.model.predict(X_test)
    test_predictions_scaled = test_predictions_scaled.flatten()
    
    # Inverse scale the predictions and actual values
    test_predictions = predictor.inverse_scale_prediction(test_predictions_scaled)
    y_test_original = predictor.inverse_scale_prediction(y_test)
    
    # Compute Evaluation Metrics
    mae = mean_absolute_error(y_test_original, test_predictions)
    mse = mean_squared_error(y_test_original, test_predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test_original, test_predictions)
    
    print("\nEvaluation Metrics on Test Set:")
    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    print(f"Mean Squared Error (MSE): {mse:.4f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
    print(f"R-squared (R²): {r2:.4f}")
    
    # Plot the predictions vs actual values
    predictor.plot_predictions(y_test_original, test_predictions)
