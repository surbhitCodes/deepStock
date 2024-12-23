import numpy as np
import pandas as pd
from pymongo import MongoClient
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
import torch.optim as optim
from transformers.models.align.convert_align_tf_to_hf import preprocess

from deepStock.news_scraper import NewsScraper
from transformers import pipeline

from test import sentiment_analysis


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
        self.sentiment_analyzer = pipeline('sentiment-analysis')
        self.historical_data = pd.read_csv(self.history_filepath)
        self.db = MongoClient(db_host, db_port)[db_name]
        self.news_collection = self.db[collection_name]
        self.preProcess()  # optional, can be commented out

    def preProcess(self):
        """
        preprocess stock price data
        :return: nothing
        """
        self.historical_data['date'] = pd.to_datetime(self.historical_data['date'])
        self.historical_data = self.historical_data.sort_values(by='date')
        scaler = MinMaxScaler(feature_range=(0, 1))
        price_features = ['Open', 'High', 'Low', 'Close', 'Volume']
        self.historical_data[price_features] = scaler.fit_transform(
            self.historical_data[price_features])  # fit transform

    def getSentimentScore(self, date):
        """
        :param date: date for sentiment analysis
        :return: Total score of the sentiment for that day
        """
        start_date = date - pd.Timedelta(days=1)
        end_date = date + pd.Timedelta(days=1)
        articles = self.news_collection.find(
            {
                'news_heading': {'$regex': self.stock_name, '$options': 'i'},
                'date': {'$gte': start_date.strf, '$lte': end_date}

            }
        )

        sentiment_score = 0
        count = 0
        for article in articles:
            sentiment = self.sentiment_analyzer(article['news_heading'])[0]
            label = sentiment['label']
            score = sentiment['score']
            
            # +score for POSITIVE and -score for NEGATIVE
            if label.upper()=='POSITIVE':
                sentiment_score+=score   
            else: 
                sentiment_score-=score
            count+=1
        
        if count == 0: return 0.0
        else: return sentiment_score/count
            


    def createDataset(self):
         """
        Create features (X) and targets (y).
        Using [Open, High, Low, Close, Volume, sentiment_score] as inputs
        and predict the next day's Close
        
        :return: numpy arrays: X, y
        """
        X = []
        y = []

        for i in range(1, len(self.historical_data)-1):
            row = self.historical_data['date'].iloc[i]
            date=row['date']

            price_features = row[['Open', 'High', 'Low', 'Close', 'Volume']].values

            news_sentiment: None = self.getSentimentScore(date)

            combined_features = np.append(price_features, news_sentiment)
            X.append(combined_features)
            next_day_close = self.historical_data.iloc[i+1]['Close']
            y.append(next_day_close)
            
        X = np.array(X, dtype=np.float32)
        y = np.array(y, dtype=np.float32).reshape(-1,1)
        return X, y



    def trainModel(self, test_size=0.2, epochs=10, lr=1e-3):
        """
        Train a simple neural network on historical data.
        :param test_size: fraction of data for validation
        :param epochs: number of epochs
        :param lr: learning rate
        """
        # create dataset
        X, y = self.createDataset()
        
        # split dataset
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=test_size, shuffle=False)
        
        # convert to torch sensors
        X_train_t = torch.from_numpy(X_train)
        y_train_t = torch.from_numpy(y_train)
        X_val_t = torch.from_numpy(X_val)
        y_val_t = torch.from_numpy(y_val)
        
        # Intitialized model, loss, optimizer
        input_dim = X_train.shape[1] # should be 6 if we have 5 price features + 1 sentiment
        self.model = SimpleNN(input_dim=input_dim, hidden_dim=32, output_dim=1)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameter, lr=lr)
        
        # Training loop
        for epoch in range(epochs):
            self.model.train()
            optimizer.zero_grad()
            
            outputs = self.model(X_train_t)
            loss.backward()
            optimizer.step()
            
            self.model.eval()
            with torch.no_grad():
                val_outputs = self.model(X_val_t)
                val_loss = criterion(val_outputs, y_val_t)
                
            # print(f'Epoch: [{epoch+1}/{epochs}],'
            #       f'Train loss: {loss.item():.6f},'
            #       f'Val loss': {val_loss.item():.6f})
            

    def latestData(self):
        """
        Return the most recent row of scaled data + sentiment
        for running predict()
        """
        last_row = self.historical_data.iloc[-1]
        date = last_row['row']
        
        price_features = last_row[['Open', 'High', 'Low', 'Close', 'Volume']].values
        
        new_sentiment = self.getSentimentScore(date)
        combined_features = np.append(price_features, news_sentiment).reshape(1, -1).astype(np.float32)
        
        return combined_score
        

    def predict(self, data):
        """
        main function to initiate prediction after training
        :rtype: object
        """
        if self.model is None: raise ValueError("Model has not been trained yet...")
        
        self. model.eval()
        
        if isinstance(data, np.ndarray):
            data = torch.from_numpy(data)

        with torch.no_grad():
            preds = self.model(data)
        
        # Convert predictions back to numpy
        return preds.numpy()


if '__name__' == '__main__':
     # Update DB with the latest news
    NewsScraper().update_news()  

    # Instantiate predictor
    predictor = StockPredict('sbi', 'sbi.csv')

    # Train model
    predictor.trainModel()

    # Make prediction on the latest data
    latest_features = predictor.latestData()
    predictions = predictor.predict(latest_features)

    # Save predictions
    pd.DataFrame(predictions, columns=['PredictedClose']).to_csv('predictions.csv', index=False)

    print(f'Predictions: {predictions}')

