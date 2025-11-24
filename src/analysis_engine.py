"""
Analysis Engine - Lightweight OOP implementation for financial analysis
"""

import pandas as pd
import numpy as np
from textblob import TextBlob
from scipy.stats import pearsonr, spearmanr
import warnings
warnings.filterwarnings('ignore')

class FinancialNewsAnalyzer:
    """OOP class for financial news sentiment analysis"""
    
    def __init__(self, news_data=None, stock_data=None):
        self.news_data = news_data
        self.stock_data = stock_data
        self.merged_data = None
        self.correlation_results = {}
        
    def load_news_data(self, filepath):
        """Load and preprocess news data"""
        self.news_data = pd.read_csv(filepath)
        self.news_data['date'] = pd.to_datetime(self.news_data['date'], format='mixed', utc=True)
        self.news_data['date_only'] = self.news_data['date'].dt.date
        return self.news_data
    
    def load_stock_data(self, symbol, data_dir):
        """Load and preprocess stock data"""
        filepath = f"{data_dir}/{symbol}.csv"
        self.stock_data = pd.read_csv(filepath)
        self.stock_data['Date'] = pd.to_datetime(self.stock_data['Date'])
        self.stock_data['date_only'] = self.stock_data['Date'].dt.date
        self.stock_data['daily_return'] = self.stock_data['Close'].pct_change()
        return self.stock_data
    
    def analyze_sentiment(self, text):
        """Analyze sentiment with comprehensive error handling"""
        if not isinstance(text, str) or pd.isna(text):
            return 0.0
        try:
            return TextBlob(text).sentiment.polarity
        except Exception as e:
            print(f"Sentiment analysis error: {e}")
            return 0.0
    
    def calculate_daily_sentiment(self):
        """Calculate daily aggregated sentiment"""
        if self.news_data is None:
            raise ValueError("News data not loaded")
        
        self.news_data['sentiment_score'] = self.news_data['headline'].apply(self.analyze_sentiment)
        
        daily_sentiment = self.news_data.groupby('date_only').agg({
            'sentiment_score': ['mean', 'std', 'count']
        }).round(4)
        
        daily_sentiment.columns = ['sentiment_mean', 'sentiment_std', 'article_count']
        return daily_sentiment.reset_index()
    
    def merge_datasets(self, symbol='AAPL'):
        """Merge news and stock data with alignment"""
        daily_sentiment = self.calculate_daily_sentiment()
        
        # Filter for specific stock
        stock_news = self.news_data[self.news_data['stock'] == symbol]
        daily_sentiment_stock = stock_news.groupby('date_only').agg({
            'sentiment_score': ['mean', 'std', 'count']
        }).round(4)
        daily_sentiment_stock.columns = ['sentiment_mean', 'sentiment_std', 'article_count']
        
        # Merge with stock data
        self.merged_data = pd.merge(
            daily_sentiment_stock.reset_index(),
            self.stock_data[['date_only', 'Close', 'daily_return']],
            on='date_only',
            how='inner'
        )
        return self.merged_data
    
    def calculate_correlations(self):
        """Calculate comprehensive correlation analysis"""
        if self.merged_data is None:
            self.merge_datasets()
        
        # Pearson correlation
        pearson_corr, pearson_p = pearsonr(
            self.merged_data['sentiment_mean'].dropna(),
            self.merged_data['daily_return'].dropna()
        )
        
        # Spearman correlation
        spearman_corr, spearman_p = spearmanr(
            self.merged_data['sentiment_mean'].dropna(),
            self.merged_data['daily_return'].dropna()
        )
        
        self.correlation_results = {
            'pearson': {'correlation': pearson_corr, 'p_value': pearson_p},
            'spearman': {'correlation': spearman_corr, 'p_value': spearman_p},
            'sample_size': len(self.merged_data),
            'date_range': f"{self.merged_data['date_only'].min()} to {self.merged_data['date_only'].max()}"
        }
        
        return self.correlation_results
    
    def generate_report(self):
        """Generate comprehensive analysis report"""
        if not self.correlation_results:
            self.calculate_correlations()
        
        report = {
            'correlation_summary': self.correlation_results,
            'sentiment_stats': self.merged_data['sentiment_mean'].describe(),
            'return_stats': self.merged_data['daily_return'].describe(),
            'data_quality': {
                'merged_records': len(self.merged_data),
                'news_articles': len(self.news_data),
                'stock_days': len(self.stock_data)
            }
        }
        return report

class TechnicalAnalyzer:
    """OOP class for technical analysis"""
    
    def __init__(self, stock_data):
        self.stock_data = stock_data
        self.indicators = {}
    
    def calculate_sma(self, window=20):
        """Calculate Simple Moving Average"""
        self.indicators[f'SMA_{window}'] = self.stock_data['Close'].rolling(window=window).mean()
        return self.indicators[f'SMA_{window}']
    
    def calculate_rsi(self, window=14):
        """Calculate Relative Strength Index"""
        delta = self.stock_data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        self.indicators['RSI'] = 100 - (100 / (1 + rs))
        return self.indicators['RSI']
    
    def get_trading_signals(self):
        """Generate trading signals based on technical indicators"""
        signals = {}
        
        if 'RSI' in self.indicators:
            current_rsi = self.indicators['RSI'].iloc[-1]
            if current_rsi < 30:
                signals['rsi'] = 'OVERSOLD'
            elif current_rsi > 70:
                signals['rsi'] = 'OVERBOUGHT'
            else:
                signals['rsi'] = 'NEUTRAL'
        
        return signals
    
    '''
    """
Financial News Sentiment Analysis Engine

A lightweight OOP implementation for analyzing correlations between financial news 
sentiment and stock price movements. Provides modular, reusable classes for 
sentiment analysis, technical analysis, and correlation calculations.

Classes:
    FinancialNewsAnalyzer: Main class for news sentiment analysis and correlation
    TechnicalAnalyzer: Technical indicator calculations and trading signals

Usage:
    >>> from analysis_engine import FinancialNewsAnalyzer, TechnicalAnalyzer
    >>> analyzer = FinancialNewsAnalyzer()
    >>> analyzer.load_news_data('path/to/news.csv')
    >>> correlations = analyzer.calculate_correlations()

Author: [Your Name]
Version: 1.0.0
Date: [Current Date]
"""

import pandas as pd
import numpy as np
from textblob import TextBlob
from scipy.stats import pearsonr, spearmanr
import warnings
warnings.filterwarnings('ignore')

class FinancialNewsAnalyzer:
    """
    Analyzes financial news sentiment and correlates with stock price movements.
    
    This class provides end-to-end analysis of financial news data, including
    sentiment scoring, data alignment with stock prices, and statistical
    correlation analysis.
    
    Attributes:
        news_data (pd.DataFrame): Loaded and processed news data
        stock_data (pd.DataFrame): Loaded and processed stock data  
        merged_data (pd.DataFrame): Aligned news and stock data
        correlation_results (dict): Statistical correlation results
        
    Example:
        >>> analyzer = FinancialNewsAnalyzer()
        >>> analyzer.load_news_data('../data/raw_analyst_ratings.csv')
        >>> analyzer.load_stock_data('AAPL', '../data/Data/')
        >>> correlations = analyzer.calculate_correlations()
        >>> print(f"Pearson correlation: {correlations['pearson']['correlation']:.4f}")
    """
    
    def __init__(self, news_data=None, stock_data=None):
        """
        Initialize the FinancialNewsAnalyzer.
        
        Args:
            news_data (pd.DataFrame, optional): Pre-loaded news data. Defaults to None.
            stock_data (pd.DataFrame, optional): Pre-loaded stock data. Defaults to None.
        """
        self.news_data = news_data
        self.stock_data = stock_data
        self.merged_data = None
        self.correlation_results = {}
    
    def load_news_data(self, filepath):
        """
        Load and preprocess financial news data from CSV file.
        
        Performs data cleaning, date parsing, and column standardization
        to prepare news data for sentiment analysis.
        
        Args:
            filepath (str): Path to the news data CSV file
            
        Returns:
            pd.DataFrame: Processed news data with datetime formatting
            
        Raises:
            FileNotFoundError: If the specified file path does not exist
            pd.errors.EmptyDataError: If the CSV file is empty
            
        Example:
            >>> analyzer.load_news_data('../data/raw_analyst_ratings.csv')
        """
        self.news_data = pd.read_csv(filepath)
        self.news_data['date'] = pd.to_datetime(self.news_data['date'], format='mixed', utc=True)
        self.news_data['date_only'] = self.news_data['date'].dt.date
        return self.news_data
    
    def load_stock_data(self, symbol, data_dir):
        """
        Load and preprocess stock price data for specified symbol.
        
        Loads stock data from CSV file, processes dates, and calculates
        daily returns for correlation analysis.
        
        Args:
            symbol (str): Stock ticker symbol (e.g., 'AAPL', 'GOOG')
            data_dir (str): Directory containing stock data files
            
        Returns:
            pd.DataFrame: Processed stock data with returns calculated
            
        Raises:
            FileNotFoundError: If stock data file doesn't exist
            KeyError: If required columns are missing
            
        Example:
            >>> analyzer.load_stock_data('AAPL', '../data/Data/')
        """
        filepath = f"{data_dir}/{symbol}.csv"
        self.stock_data = pd.read_csv(filepath)
        self.stock_data['Date'] = pd.to_datetime(self.stock_data['Date'])
        self.stock_data['date_only'] = self.stock_data['Date'].dt.date
        self.stock_data['daily_return'] = self.stock_data['Close'].pct_change()
        return self.stock_data
    
    def analyze_sentiment(self, text):
        """
        Analyze sentiment of financial headline using TextBlob.
        
        Converts text to sentiment polarity score ranging from -1 (negative)
        to +1 (positive). Includes comprehensive error handling for edge cases.
        
        Args:
            text (str): Financial news headline to analyze
            
        Returns:
            float: Sentiment polarity score between -1.0 and 1.0
            
        Example:
            >>> score = analyzer.analyze_sentiment("Apple stock surges on strong earnings")
            >>> print(f"Sentiment score: {score:.3f}")  # Output: ~0.5
        """
        if not isinstance(text, str) or pd.isna(text):
            return 0.0
        try:
            return TextBlob(text).sentiment.polarity
        except Exception as e:
            print(f"Sentiment analysis error: {e}")
            return 0.0
    
    def calculate_daily_sentiment(self):
        """
        Calculate daily aggregated sentiment scores from news headlines.
        
        Processes all news articles to compute daily average sentiment,
        standard deviation, and article count for correlation analysis.
        
        Returns:
            pd.DataFrame: Daily sentiment statistics with columns:
                - sentiment_mean: Average daily sentiment score
                - sentiment_std: Standard deviation of sentiment scores
                - article_count: Number of articles per day
                
        Raises:
            ValueError: If news data hasn't been loaded
            
        Example:
            >>> daily_sentiment = analyzer.calculate_daily_sentiment()
            >>> print(f"Average daily sentiment: {daily_sentiment['sentiment_mean'].mean():.3f}")
        """
        if self.news_data is None:
            raise ValueError("News data not loaded. Call load_news_data() first.")
        
        self.news_data['sentiment_score'] = self.news_data['headline'].apply(self.analyze_sentiment)
        
        daily_sentiment = self.news_data.groupby('date_only').agg({
            'sentiment_score': ['mean', 'std', 'count']
        }).round(4)
        
        daily_sentiment.columns = ['sentiment_mean', 'sentiment_std', 'article_count']
        return daily_sentiment.reset_index()
    
    def merge_datasets(self, symbol='AAPL'):
        """
        Merge news sentiment data with corresponding stock price data.
        
        Aligns daily sentiment scores with stock returns by date and filters
        for a specific stock symbol to ensure proper correlation analysis.
        
        Args:
            symbol (str, optional): Stock symbol to filter. Defaults to 'AAPL'.
            
        Returns:
            pd.DataFrame: Merged dataset with sentiment and return data
            
        Example:
            >>> merged_data = analyzer.merge_datasets('AAPL')
            >>> print(f"Merged {len(merged_data)} days of data")
        """
        daily_sentiment = self.calculate_daily_sentiment()
        
        # Filter for specific stock
        stock_news = self.news_data[self.news_data['stock'] == symbol]
        daily_sentiment_stock = stock_news.groupby('date_only').agg({
            'sentiment_score': ['mean', 'std', 'count']
        }).round(4)
        daily_sentiment_stock.columns = ['sentiment_mean', 'sentiment_std', 'article_count']
        
        # Merge with stock data
        self.merged_data = pd.merge(
            daily_sentiment_stock.reset_index(),
            self.stock_data[['date_only', 'Close', 'daily_return']],
            on='date_only',
            how='inner'
        )
        return self.merged_data
    
    def calculate_correlations(self):
        """
        Calculate statistical correlations between sentiment and stock returns.
        
        Computes both Pearson and Spearman correlation coefficients with
        p-values to assess relationship significance.
        
        Returns:
            dict: Correlation results containing:
                - pearson: Pearson correlation coefficient and p-value
                - spearman: Spearman rank correlation and p-value
                - sample_size: Number of data points used
                - date_range: Analysis period
                
        Example:
            >>> correlations = analyzer.calculate_correlations()
            >>> print(f"Pearson r: {correlations['pearson']['correlation']:.4f}")
            >>> print(f"P-value: {correlations['pearson']['p_value']:.4f}")
        """
        if self.merged_data is None:
            self.merge_datasets()
        
        # Pearson correlation
        pearson_corr, pearson_p = pearsonr(
            self.merged_data['sentiment_mean'].dropna(),
            self.merged_data['daily_return'].dropna()
        )
        
        # Spearman correlation
        spearman_corr, spearman_p = spearmanr(
            self.merged_data['sentiment_mean'].dropna(),
            self.merged_data['daily_return'].dropna()
        )
        
        self.correlation_results = {
            'pearson': {'correlation': pearson_corr, 'p_value': pearson_p},
            'spearman': {'correlation': spearman_corr, 'p_value': spearman_p},
            'sample_size': len(self.merged_data),
            'date_range': f"{self.merged_data['date_only'].min()} to {self.merged_data['date_only'].max()}"
        }
        
        return self.correlation_results
    
    def generate_report(self):
        """
        Generate comprehensive analysis report with key findings.
        
        Compiles correlation results, descriptive statistics, and data quality
        metrics into a structured report for business decision-making.
        
        Returns:
            dict: Comprehensive report containing:
                - correlation_summary: Statistical correlation results
                - sentiment_stats: Descriptive statistics of sentiment scores
                - return_stats: Descriptive statistics of stock returns
                - data_quality: Dataset size and coverage information
                
        Example:
            >>> report = analyzer.generate_report()
            >>> print(f"Sample size: {report['data_quality']['merged_records']}")
            >>> print(f"Average return: {report['return_stats']['mean']:.4f}")
        """
        if not self.correlation_results:
            self.calculate_correlations()
        
        report = {
            'correlation_summary': self.correlation_results,
            'sentiment_stats': self.merged_data['sentiment_mean'].describe(),
            'return_stats': self.merged_data['daily_return'].describe(),
            'data_quality': {
                'merged_records': len(self.merged_data),
                'news_articles': len(self.news_data),
                'stock_days': len(self.stock_data)
            }
        }
        return report

class TechnicalAnalyzer:
    """
    Provides technical analysis capabilities for stock price data.
    
    Calculates various technical indicators and generates trading signals
    based on momentum, trend, and volatility metrics.
    
    Attributes:
        stock_data (pd.DataFrame): Stock price data for analysis
        indicators (dict): Calculated technical indicators
        
    Example:
        >>> analyzer = TechnicalAnalyzer(stock_data)
        >>> analyzer.calculate_rsi(14)
        >>> signals = analyzer.get_trading_signals()
        >>> print(f"RSI Signal: {signals['rsi']}")
    """
    
    def __init__(self, stock_data):
        """
        Initialize TechnicalAnalyzer with stock price data.
        
        Args:
            stock_data (pd.DataFrame): Stock data with OHLCV columns
        """
        self.stock_data = stock_data
        self.indicators = {}
    
    def calculate_sma(self, window=20):
        """
        Calculate Simple Moving Average for stock prices.
        
        Computes rolling mean of closing prices to identify trend direction
        and potential support/resistance levels.
        
        Args:
            window (int, optional): Rolling window size. Defaults to 20.
            
        Returns:
            pd.Series: Simple Moving Average values
            
        Example:
            >>> sma_20 = analyzer.calculate_sma(20)
            >>> print(f"Latest SMA: {sma_20.iloc[-1]:.2f}")
        """
        self.indicators[f'SMA_{window}'] = self.stock_data['Close'].rolling(window=window).mean()
        return self.indicators[f'SMA_{window}']
    
    def calculate_rsi(self, window=14):
        """
        Calculate Relative Strength Index (RSI) momentum oscillator.
        
        RSI measures the speed and change of price movements on a scale
        of 0-100. Values above 70 indicate overbought conditions,
        below 30 indicate oversold conditions.
        
        Args:
            window (int, optional): Lookback period. Defaults to 14.
            
        Returns:
            pd.Series: RSI values between 0 and 100
            
        Example:
            >>> rsi = analyzer.calculate_rsi(14)
            >>> print(f"Current RSI: {rsi.iloc[-1]:.1f}")
        """
        delta = self.stock_data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        self.indicators['RSI'] = 100 - (100 / (1 + rs))
        return self.indicators['RSI']
    
    def get_trading_signals(self):
        """
        Generate trading signals based on technical indicators.
        
        Analyzes current indicator values to produce actionable trading
        signals such as overbought/oversold conditions.
        
        Returns:
            dict: Trading signals with indicator-based recommendations
            
        Example:
            >>> signals = analyzer.get_trading_signals()
            >>> if signals['rsi'] == 'OVERSOLD':
            ...     print("Potential buying opportunity")
        """
        signals = {}
        
        if 'RSI' in self.indicators:
            current_rsi = self.indicators['RSI'].iloc[-1]
            if current_rsi < 30:
                signals['rsi'] = 'OVERSOLD'
            elif current_rsi > 70:
                signals['rsi'] = 'OVERBOUGHT'
            else:
                signals['rsi'] = 'NEUTRAL'
        
        return signals
    
    
    '''