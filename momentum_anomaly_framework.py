"""
Momentum Anomaly Testing Framework

This module provides a flexible framework for testing momentum anomalies with:
- Different lag periods (day, week, month)
- Multiple period combinations (e.g., 6+3+1 months)
- No moving averages approach
- Portfolio construction and backtesting capabilities

Author: AI Assistant
Date: 2024
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional, Union
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class MomentumAnomalyTester:
    """
    A flexible framework for testing momentum anomalies with different lags and period combinations.
    """
    
    def __init__(self, data: pd.DataFrame, price_col: str = 'close', date_col: str = 'date'):
        """
        Initialize the momentum tester.
        
        Parameters:
        -----------
        data : pd.DataFrame
            DataFrame with price data
        price_col : str
            Column name for price data
        date_col : str
            Column name for date data
        """
        self.data = data.copy()
        self.price_col = price_col
        self.date_col = date_col
        
        # Ensure date column is datetime
        self.data[date_col] = pd.to_datetime(self.data[date_col])
        self.data = self.data.sort_values(date_col).reset_index(drop=True)
        
        # Calculate returns
        self._calculate_returns()
    
    def _calculate_returns(self):
        """Calculate various return measures."""
        self.data['returns'] = self.data[self.price_col].pct_change()
        self.data['log_returns'] = np.log(self.data[self.price_col] / self.data[self.price_col].shift(1))
    
    def calculate_momentum_lags(self, lags: List[int], lag_type: str = 'month') -> pd.DataFrame:
        """
        Calculate momentum for different lag periods.
        
        Parameters:
        -----------
        lags : List[int]
            List of lag periods to calculate
        lag_type : str
            Type of lag ('day', 'week', 'month')
        
        Returns:
        --------
        pd.DataFrame
            DataFrame with momentum calculations
        """
        result_df = self.data.copy()
        
        for lag in lags:
            if lag_type == 'day':
                # Daily momentum
                result_df[f'momentum_{lag}d'] = self.data[self.price_col].pct_change(lag)
                result_df[f'log_momentum_{lag}d'] = np.log(self.data[self.price_col] / self.data[self.price_col].shift(lag))
                
            elif lag_type == 'week':
                # Weekly momentum (assuming 5 trading days per week)
                days_lag = lag * 5
                result_df[f'momentum_{lag}w'] = self.data[self.price_col].pct_change(days_lag)
                result_df[f'log_momentum_{lag}w'] = np.log(self.data[self.price_col] / self.data[self.price_col].shift(days_lag))
                
            elif lag_type == 'month':
                # Monthly momentum (assuming 21 trading days per month)
                days_lag = lag * 21
                result_df[f'momentum_{lag}m'] = self.data[self.price_col].pct_change(days_lag)
                result_df[f'log_momentum_{lag}m'] = np.log(self.data[self.price_col] / self.data[self.price_col].shift(days_lag))
        
        return result_df
    
    def calculate_combined_momentum(self, periods: List[int], lag_type: str = 'month', 
                                  method: str = 'sum', min_positive: int = None) -> pd.DataFrame:
        """
        Calculate combined momentum across multiple periods.
        
        Parameters:
        -----------
        periods : List[int]
            List of periods to combine (e.g., [6, 3, 1] for 6+3+1 month strategy)
        lag_type : str
            Type of lag ('day', 'week', 'month')
        method : str
            Combination method ('sum', 'mean', 'product')
        min_positive : int, optional
            Minimum number of positive periods required (e.g., 2 for 6+3+1 means at least 2 must be positive)
        
        Returns:
        --------
        pd.DataFrame
            DataFrame with combined momentum calculations
        """
        # Calculate individual momentum measures
        momentum_df = self.calculate_momentum_lags(periods, lag_type)
        
        # Get momentum column names
        momentum_cols = [col for col in momentum_df.columns if col.startswith('momentum_') and not col.startswith('log_')]
        
        # Calculate combined momentum
        if method == 'sum':
            momentum_df['combined_momentum'] = momentum_df[momentum_cols].sum(axis=1)
        elif method == 'mean':
            momentum_df['combined_momentum'] = momentum_df[momentum_cols].mean(axis=1)
        elif method == 'product':
            momentum_df['combined_momentum'] = momentum_df[momentum_cols].product(axis=1)
        
        # Apply minimum positive requirement
        if min_positive is not None:
            positive_count = (momentum_df[momentum_cols] > 0).sum(axis=1)
            momentum_df['combined_momentum'] = np.where(
                positive_count >= min_positive,
                momentum_df['combined_momentum'],
                0
            )
        
        return momentum_df
    
    def create_momentum_portfolio(self, momentum_col: str, n_assets: int = 10, 
                                method: str = 'top_bottom') -> pd.DataFrame:
        """
        Create momentum portfolios based on momentum signals.
        
        Parameters:
        -----------
        momentum_col : str
            Column name for momentum signal
        n_assets : int
            Number of assets to include in each portfolio
        method : str
            Portfolio method ('top_bottom', 'quintiles', 'deciles')
        
        Returns:
        --------
        pd.DataFrame
            DataFrame with portfolio assignments
        """
        portfolio_df = self.data.copy()
        
        if method == 'top_bottom':
            # Top and bottom performers
            portfolio_df['portfolio'] = 'neutral'
            portfolio_df.loc[portfolio_df[momentum_col] > portfolio_df[momentum_col].quantile(0.9), 'portfolio'] = 'winner'
            portfolio_df.loc[portfolio_df[momentum_col] < portfolio_df[momentum_col].quantile(0.1), 'portfolio'] = 'loser'
            
        elif method == 'quintiles':
            # Quintile portfolios
            portfolio_df['quintile'] = pd.qcut(portfolio_df[momentum_col], 5, labels=['Q1', 'Q2', 'Q3', 'Q4', 'Q5'])
            portfolio_df['portfolio'] = portfolio_df['quintile'].map({
                'Q1': 'loser', 'Q2': 'low', 'Q3': 'neutral', 'Q4': 'high', 'Q5': 'winner'
            })
            
        elif method == 'deciles':
            # Decile portfolios
            portfolio_df['decile'] = pd.qcut(portfolio_df[momentum_col], 10, labels=[f'D{i}' for i in range(1, 11)])
            portfolio_df['portfolio'] = portfolio_df['decile'].map({
                'D1': 'loser', 'D2': 'low', 'D3': 'neutral', 'D4': 'neutral',
                'D5': 'neutral', 'D6': 'neutral', 'D7': 'neutral', 'D8': 'neutral',
                'D9': 'high', 'D10': 'winner'
            })
        
        return portfolio_df
    
    def calculate_portfolio_returns(self, portfolio_df: pd.DataFrame, 
                                  portfolio_col: str = 'portfolio') -> pd.DataFrame:
        """
        Calculate portfolio returns for different momentum portfolios.
        
        Parameters:
        -----------
        portfolio_df : pd.DataFrame
            DataFrame with portfolio assignments
        portfolio_col : str
            Column name for portfolio assignments
        
        Returns:
        --------
        pd.DataFrame
            DataFrame with portfolio returns
        """
        portfolio_returns = portfolio_df.groupby([self.date_col, portfolio_col])['returns'].mean().reset_index()
        
        # Pivot to get portfolios as columns
        portfolio_pivot = portfolio_returns.pivot(index=self.date_col, 
                                               columns=portfolio_col, 
                                               values='returns')
        
        # Calculate cumulative returns
        portfolio_pivot = portfolio_pivot.fillna(0)
        cumulative_returns = (1 + portfolio_pivot).cumprod()
        
        return cumulative_returns
    
    def calculate_momentum_statistics(self, momentum_col: str) -> Dict:
        """
        Calculate momentum statistics.
        
        Parameters:
        -----------
        momentum_col : str
            Column name for momentum signal
        
        Returns:
        --------
        Dict
            Dictionary with momentum statistics
        """
        momentum_data = self.data[momentum_col].dropna()
        
        stats = {
            'mean': momentum_data.mean(),
            'std': momentum_data.std(),
            'skewness': momentum_data.skew(),
            'kurtosis': momentum_data.kurtosis(),
            'autocorrelation': momentum_data.autocorr(),
            'positive_ratio': (momentum_data > 0).mean(),
            'negative_ratio': (momentum_data < 0).mean(),
            'zero_ratio': (momentum_data == 0).mean()
        }
        
        return stats
    
    def test_momentum_strategy(self, periods: List[int], lag_type: str = 'month',
                              method: str = 'sum', min_positive: int = None,
                              portfolio_method: str = 'top_bottom') -> Dict:
        """
        Comprehensive momentum strategy test.
        
        Parameters:
        -----------
        periods : List[int]
            List of periods to combine
        lag_type : str
            Type of lag ('day', 'week', 'month')
        method : str
            Combination method ('sum', 'mean', 'product')
        min_positive : int, optional
            Minimum number of positive periods required
        portfolio_method : str
            Portfolio construction method
        
        Returns:
        --------
        Dict
            Dictionary with strategy results
        """
        # Calculate combined momentum
        momentum_df = self.calculate_combined_momentum(periods, lag_type, method, min_positive)
        
        # Create portfolios
        portfolio_df = self.create_momentum_portfolio('combined_momentum', method=portfolio_method)
        
        # Calculate portfolio returns
        portfolio_returns = self.calculate_portfolio_returns(portfolio_df)
        
        # Calculate statistics
        momentum_stats = self.calculate_momentum_statistics('combined_momentum')
        
        # Calculate strategy performance
        if 'winner' in portfolio_returns.columns and 'loser' in portfolio_returns.columns:
            winner_loser_spread = portfolio_returns['winner'] - portfolio_returns['loser']
            spread_stats = {
                'final_spread': winner_loser_spread.iloc[-1],
                'max_spread': winner_loser_spread.max(),
                'min_spread': winner_loser_spread.min(),
                'spread_volatility': winner_loser_spread.std()
            }
        else:
            spread_stats = {}
        
        results = {
            'momentum_stats': momentum_stats,
            'portfolio_returns': portfolio_returns,
            'spread_stats': spread_stats,
            'parameters': {
                'periods': periods,
                'lag_type': lag_type,
                'method': method,
                'min_positive': min_positive,
                'portfolio_method': portfolio_method
            }
        }
        
        return results

def create_sample_data(n_days: int = 1000, n_assets: int = 100) -> pd.DataFrame:
    """
    Create sample data for testing momentum strategies.
    
    Parameters:
    -----------
    n_days : int
        Number of days of data
    n_assets : int
        Number of assets
    
    Returns:
    --------
    pd.DataFrame
        Sample price data
    """
    np.random.seed(42)
    
    # Generate dates
    start_date = datetime(2020, 1, 1)
    dates = [start_date + timedelta(days=i) for i in range(n_days)]
    
    # Generate sample data
    data = []
    for asset_id in range(n_assets):
        # Generate random walk with momentum
        prices = [100]  # Starting price
        for i in range(1, n_days):
            # Add some momentum effect
            momentum_effect = 0.001 * (prices[-1] - prices[max(0, i-20)]) / prices[max(0, i-20)]
            random_return = np.random.normal(0, 0.02) + momentum_effect
            new_price = prices[-1] * (1 + random_return)
            prices.append(new_price)
        
        asset_data = pd.DataFrame({
            'asset_id': asset_id,
            'date': dates,
            'close': prices
        })
        data.append(asset_data)
    
    return pd.concat(data, ignore_index=True)

# Example usage and testing
if __name__ == "__main__":
    # Create sample data
    print("Creating sample data...")
    sample_data = create_sample_data(n_days=500, n_assets=50)
    
    # Initialize tester
    tester = MomentumAnomalyTester(sample_data, price_col='close', date_col='date')
    
    # Test different momentum strategies
    print("\nTesting momentum strategies...")
    
    # Strategy 1: Simple 6-month momentum
    print("\n1. Testing 6-month momentum:")
    results_6m = tester.test_momentum_strategy(periods=[6], lag_type='month')
    print(f"6-month momentum mean: {results_6m['momentum_stats']['mean']:.4f}")
    print(f"6-month momentum std: {results_6m['momentum_stats']['std']:.4f}")
    
    # Strategy 2: Combined 6+3+1 month momentum
    print("\n2. Testing 6+3+1 month momentum:")
    results_combined = tester.test_momentum_strategy(
        periods=[6, 3, 1], 
        lag_type='month', 
        method='sum',
        min_positive=2  # At least 2 periods must be positive
    )
    print(f"Combined momentum mean: {results_combined['momentum_stats']['mean']:.4f}")
    print(f"Positive ratio: {results_combined['momentum_stats']['positive_ratio']:.4f}")
    
    # Strategy 3: Weekly momentum
    print("\n3. Testing weekly momentum:")
    results_weekly = tester.test_momentum_strategy(
        periods=[4, 2, 1], 
        lag_type='week', 
        method='sum'
    )
    print(f"Weekly momentum mean: {results_weekly['momentum_stats']['mean']:.4f}")
    
    print("\nMomentum testing framework ready for use!")