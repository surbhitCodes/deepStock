import numpy as np
import pandas as pd
from pymongo import MongoClient
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
import torch.optim as optim
from scraper import NewsScraper
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

        # preprocess stock price data (optional)
        self.historical_data['date'] = pd.to_datetime(self.historical_data['date'])
        self.historical_data = self.historical_data.sort_values(by='date')
        scaler = MinMaxScaler(feature_range=(0, 1))
        price_features = ['Open', 'High', 'Low', 'Close', 'Volume']
        self.historical_data[price_features] = scaler.fit_transform(
            self.historical_data[price_features])  # fit transform

    def get_sentiment_score(self, stock_name, date):
        """
        :param stock_name: name of the stock
        :param date: date for sentiment analysis
        :return: Total score of the sentiment for that day
        """
        start_date = date - pd.Timedelta(days=1)
        end_date = date + pd.Timedelta(days=1)
        articles = self.news_collection.find(
            {
                'news_heading': {'$regex': stock_name, '$options': 'i'},
                'date': {'$gte': start_date.strf, '$lte': end_date}

            }
        )


if '__name__' == '__main__':
    predict = StockPredict('sbi', 'sbi.csv')
    predict.predict()
