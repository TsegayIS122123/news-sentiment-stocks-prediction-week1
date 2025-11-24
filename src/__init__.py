"""
Financial News Sentiment Analysis Package

This package provides modules for analyzing financial news sentiment
and its correlation with stock price movements.
"""

__version__ = "1.0.0"
__author__ = "Tsegay"

print("Financial News Analysis package imported successfully")
# Import main classes for easy access
from .analysis_engine import FinancialNewsAnalyzer, TechnicalAnalyzer
from .data_loader import load_news_data, load_stock_data
from .sentiment_analyzer import analyze_sentiment, get_sentiment_label

__all__ = [
    'FinancialNewsAnalyzer',
    'TechnicalAnalyzer', 
    'load_news_data',
    'load_stock_data',
    'analyze_sentiment',
    'get_sentiment_label'
]

print("Financial News Analysis package imported successfully! ")