"""
Simplified Momentum Anomaly Testing Framework Demo

This is a demonstration version that works without external dependencies.
For full functionality, install pandas, numpy, matplotlib, and seaborn.

Author: AI Assistant
Date: 2024
"""

import random
import math
from datetime import datetime, timedelta

class SimpleMomentumTester:
    """
    A simplified momentum tester that demonstrates the framework structure.
    """
    
    def __init__(self, data):
        """
        Initialize the momentum tester.
        
        Parameters:
        -----------
        data : list
            List of dictionaries with 'date', 'close', 'asset_id' keys
        """
        self.data = data
        self.price_col = 'close'
        self.date_col = 'date'
        
    def calculate_momentum_lags(self, lags, lag_type='month'):
        """
        Calculate momentum for different lag periods.
        
        Parameters:
        -----------
        lags : list
            List of lag periods to calculate
        lag_type : str
            Type of lag ('day', 'week', 'month')
        
        Returns:
        --------
        dict
            Dictionary with momentum calculations
        """
        result = {}
        
        for lag in lags:
            if lag_type == 'day':
                days_lag = lag
            elif lag_type == 'week':
                days_lag = lag * 5
            elif lag_type == 'month':
                days_lag = lag * 21
            else:
                days_lag = lag
            
            # Calculate momentum for each asset
            momentum_values = []
            for i in range(len(self.data)):
                if i >= days_lag:
                    current_price = self.data[i]['close']
                    past_price = self.data[i - days_lag]['close']
                    momentum = (current_price - past_price) / past_price
                    momentum_values.append(momentum)
                else:
                    momentum_values.append(0)
            
            result[f'momentum_{lag}{lag_type[0]}'] = momentum_values
        
        return result
    
    def calculate_combined_momentum(self, periods, lag_type='month', 
                                  method='sum', min_positive=None):
        """
        Calculate combined momentum across multiple periods.
        
        Parameters:
        -----------
        periods : list
            List of periods to combine
        lag_type : str
            Type of lag ('day', 'week', 'month')
        method : str
            Combination method ('sum', 'mean', 'product')
        min_positive : int, optional
            Minimum number of positive periods required
        
        Returns:
        --------
        list
            List of combined momentum values
        """
        # Calculate individual momentum measures
        momentum_dict = self.calculate_momentum_lags(periods, lag_type)
        
        # Get momentum values
        momentum_lists = list(momentum_dict.values())
        
        # Calculate combined momentum
        combined_momentum = []
        for i in range(len(self.data)):
            values = [momentum_lists[j][i] for j in range(len(momentum_lists))]
            
            if method == 'sum':
                combined = sum(values)
            elif method == 'mean':
                combined = sum(values) / len(values)
            elif method == 'product':
                combined = 1
                for val in values:
                    combined *= (1 + val)
                combined -= 1
            
            # Apply minimum positive requirement
            if min_positive is not None:
                positive_count = sum(1 for val in values if val > 0)
                if positive_count < min_positive:
                    combined = 0
            
            combined_momentum.append(combined)
        
        return combined_momentum
    
    def calculate_momentum_statistics(self, momentum_values):
        """
        Calculate momentum statistics.
        
        Parameters:
        -----------
        momentum_values : list
            List of momentum values
        
        Returns:
        --------
        dict
            Dictionary with momentum statistics
        """
        # Filter out zeros (initial values)
        non_zero_values = [v for v in momentum_values if v != 0]
        
        if not non_zero_values:
            return {
                'mean': 0,
                'std': 0,
                'positive_ratio': 0,
                'negative_ratio': 0
            }
        
        mean_val = sum(non_zero_values) / len(non_zero_values)
        
        # Calculate standard deviation
        variance = sum((x - mean_val) ** 2 for x in non_zero_values) / len(non_zero_values)
        std_val = math.sqrt(variance)
        
        # Calculate ratios
        positive_count = sum(1 for x in non_zero_values if x > 0)
        negative_count = sum(1 for x in non_zero_values if x < 0)
        total_count = len(non_zero_values)
        
        return {
            'mean': mean_val,
            'std': std_val,
            'positive_ratio': positive_count / total_count if total_count > 0 else 0,
            'negative_ratio': negative_count / total_count if total_count > 0 else 0
        }
    
    def test_momentum_strategy(self, periods, lag_type='month',
                              method='sum', min_positive=None):
        """
        Comprehensive momentum strategy test.
        
        Parameters:
        -----------
        periods : list
            List of periods to combine
        lag_type : str
            Type of lag ('day', 'week', 'month')
        method : str
            Combination method ('sum', 'mean', 'product')
        min_positive : int, optional
            Minimum number of positive periods required
        
        Returns:
        --------
        dict
            Dictionary with strategy results
        """
        # Calculate combined momentum
        combined_momentum = self.calculate_combined_momentum(
            periods, lag_type, method, min_positive
        )
        
        # Calculate statistics
        momentum_stats = self.calculate_momentum_statistics(combined_momentum)
        
        results = {
            'momentum_stats': momentum_stats,
            'combined_momentum': combined_momentum,
            'parameters': {
                'periods': periods,
                'lag_type': lag_type,
                'method': method,
                'min_positive': min_positive
            }
        }
        
        return results

def create_sample_data(n_days=100, n_assets=10):
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
    list
        List of dictionaries with sample data
    """
    random.seed(42)
    
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
            if i >= 20:
                momentum_effect = 0.001 * (prices[-1] - prices[i-20]) / prices[i-20]
            else:
                momentum_effect = 0
            random_return = random.gauss(0, 0.02) + momentum_effect
            new_price = prices[-1] * (1 + random_return)
            prices.append(new_price)
        
        # Create data entries
        for i, (date, price) in enumerate(zip(dates, prices)):
            data.append({
                'asset_id': asset_id,
                'date': date,
                'close': price
            })
    
    return data

def demonstrate_framework():
    """
    Demonstrate the momentum framework functionality.
    """
    print("=== Momentum Anomaly Testing Framework Demo ===\n")
    
    # Create sample data
    print("1. Creating sample data...")
    sample_data = create_sample_data(n_days=200, n_assets=5)
    print(f"   Created {len(sample_data)} data points across 5 assets")
    
    # Initialize tester
    tester = SimpleMomentumTester(sample_data)
    
    # Test different momentum strategies
    print("\n2. Testing Momentum Strategies:")
    
    # Strategy 1: Simple 6-month momentum
    print("\n   Strategy 1: 6-month momentum")
    results_6m = tester.test_momentum_strategy(periods=[6], lag_type='month')
    stats_6m = results_6m['momentum_stats']
    print(f"      Mean: {stats_6m['mean']:.4f}")
    print(f"      Std: {stats_6m['std']:.4f}")
    print(f"      Positive ratio: {stats_6m['positive_ratio']:.4f}")
    
    # Strategy 2: Combined 6+3+1 month momentum
    print("\n   Strategy 2: 6+3+1 month momentum (sum)")
    results_6_3_1 = tester.test_momentum_strategy(
        periods=[6, 3, 1], 
        lag_type='month', 
        method='sum'
    )
    stats_6_3_1 = results_6_3_1['momentum_stats']
    print(f"      Mean: {stats_6_3_1['mean']:.4f}")
    print(f"      Std: {stats_6_3_1['std']:.4f}")
    print(f"      Positive ratio: {stats_6_3_1['positive_ratio']:.4f}")
    
    # Strategy 3: Combined momentum with minimum positive requirement
    print("\n   Strategy 3: 6+3+1 month momentum (min 2 positive)")
    results_6_3_1_min2 = tester.test_momentum_strategy(
        periods=[6, 3, 1], 
        lag_type='month', 
        method='sum',
        min_positive=2
    )
    stats_6_3_1_min2 = results_6_3_1_min2['momentum_stats']
    print(f"      Mean: {stats_6_3_1_min2['mean']:.4f}")
    print(f"      Std: {stats_6_3_1_min2['std']:.4f}")
    print(f"      Positive ratio: {stats_6_3_1_min2['positive_ratio']:.4f}")
    
    # Strategy 4: Different lag types
    print("\n   Strategy 4: Different lag types")
    
    # Daily momentum
    results_daily = tester.test_momentum_strategy(
        periods=[5, 10], 
        lag_type='day', 
        method='sum'
    )
    stats_daily = results_daily['momentum_stats']
    print(f"      Daily (5+10 days) - Mean: {stats_daily['mean']:.4f}")
    
    # Weekly momentum
    results_weekly = tester.test_momentum_strategy(
        periods=[2, 4], 
        lag_type='week', 
        method='sum'
    )
    stats_weekly = results_weekly['momentum_stats']
    print(f"      Weekly (2+4 weeks) - Mean: {stats_weekly['mean']:.4f}")
    
    # Monthly momentum
    results_monthly = tester.test_momentum_strategy(
        periods=[3, 6], 
        lag_type='month', 
        method='sum'
    )
    stats_monthly = results_monthly['momentum_stats']
    print(f"      Monthly (3+6 months) - Mean: {stats_monthly['mean']:.4f}")
    
    # Strategy 5: Different combination methods
    print("\n   Strategy 5: Different combination methods")
    
    for method in ['sum', 'mean', 'product']:
        results_method = tester.test_momentum_strategy(
            periods=[6, 3, 1], 
            lag_type='month', 
            method=method
        )
        stats_method = results_method['momentum_stats']
        print(f"      {method.capitalize()} method - Mean: {stats_method['mean']:.4f}, "
              f"Std: {stats_method['std']:.4f}")
    
    print("\n=== Framework Features Demonstrated ===")
    print("✓ Flexible lag periods (day, week, month)")
    print("✓ Multiple period combinations (e.g., 6+3+1 months)")
    print("✓ Different combination methods (sum, mean, product)")
    print("✓ Minimum positive period requirements")
    print("✓ Comprehensive statistical analysis")
    print("✓ No moving averages approach")
    
    print("\n=== Framework Ready for Research ===")
    print("To use with real data, install pandas and numpy:")
    print("pip install pandas numpy matplotlib seaborn")
    print("\nThen import the full framework:")
    print("from momentum_anomaly_framework import MomentumAnomalyTester")

if __name__ == "__main__":
    demonstrate_framework()