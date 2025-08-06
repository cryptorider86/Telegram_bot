"""
Momentum Anomaly Examples

This script demonstrates various momentum strategies using the MomentumAnomalyTester framework.
It includes examples for different lag periods, period combinations, and portfolio construction methods.

Author: AI Assistant
Date: 2024
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from momentum_anomaly_framework import MomentumAnomalyTester, create_sample_data

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def run_momentum_examples():
    """
    Run comprehensive momentum strategy examples.
    """
    print("=== Momentum Anomaly Testing Examples ===\n")
    
    # Create sample data
    print("1. Creating sample data...")
    sample_data = create_sample_data(n_days=1000, n_assets=100)
    print(f"   Created data with {len(sample_data)} observations across {sample_data['asset_id'].nunique()} assets")
    
    # Initialize tester
    tester = MomentumAnomalyTester(sample_data, price_col='close', date_col='date')
    
    # Example 1: Simple momentum strategies
    print("\n2. Testing Simple Momentum Strategies:")
    
    # 6-month momentum
    results_6m = tester.test_momentum_strategy(periods=[6], lag_type='month')
    print(f"   6-month momentum - Mean: {results_6m['momentum_stats']['mean']:.4f}, "
          f"Std: {results_6m['momentum_stats']['std']:.4f}")
    
    # 12-month momentum
    results_12m = tester.test_momentum_strategy(periods=[12], lag_type='month')
    print(f"   12-month momentum - Mean: {results_12m['momentum_stats']['mean']:.4f}, "
          f"Std: {results_12m['momentum_stats']['std']:.4f}")
    
    # Example 2: Combined momentum strategies
    print("\n3. Testing Combined Momentum Strategies:")
    
    # 6+3+1 month strategy (sum)
    results_6_3_1_sum = tester.test_momentum_strategy(
        periods=[6, 3, 1], 
        lag_type='month', 
        method='sum'
    )
    print(f"   6+3+1 month (sum) - Mean: {results_6_3_1_sum['momentum_stats']['mean']:.4f}, "
          f"Positive ratio: {results_6_3_1_sum['momentum_stats']['positive_ratio']:.4f}")
    
    # 6+3+1 month strategy with minimum positive requirement
    results_6_3_1_min2 = tester.test_momentum_strategy(
        periods=[6, 3, 1], 
        lag_type='month', 
        method='sum',
        min_positive=2  # At least 2 periods must be positive
    )
    print(f"   6+3+1 month (min 2 positive) - Mean: {results_6_3_1_min2['momentum_stats']['mean']:.4f}, "
          f"Positive ratio: {results_6_3_1_min2['momentum_stats']['positive_ratio']:.4f}")
    
    # 6+3+1 month strategy (mean)
    results_6_3_1_mean = tester.test_momentum_strategy(
        periods=[6, 3, 1], 
        lag_type='month', 
        method='mean'
    )
    print(f"   6+3+1 month (mean) - Mean: {results_6_3_1_mean['momentum_stats']['mean']:.4f}, "
          f"Std: {results_6_3_1_mean['momentum_stats']['std']:.4f}")
    
    # Example 3: Different lag types
    print("\n4. Testing Different Lag Types:")
    
    # Daily momentum
    results_daily = tester.test_momentum_strategy(
        periods=[5, 10, 20], 
        lag_type='day', 
        method='sum'
    )
    print(f"   Daily momentum (5+10+20 days) - Mean: {results_daily['momentum_stats']['mean']:.4f}")
    
    # Weekly momentum
    results_weekly = tester.test_momentum_strategy(
        periods=[4, 8, 12], 
        lag_type='week', 
        method='sum'
    )
    print(f"   Weekly momentum (4+8+12 weeks) - Mean: {results_weekly['momentum_stats']['mean']:.4f}")
    
    # Monthly momentum
    results_monthly = tester.test_momentum_strategy(
        periods=[3, 6, 12], 
        lag_type='month', 
        method='sum'
    )
    print(f"   Monthly momentum (3+6+12 months) - Mean: {results_monthly['momentum_stats']['mean']:.4f}")
    
    # Example 4: Portfolio construction methods
    print("\n5. Testing Portfolio Construction Methods:")
    
    # Top-bottom portfolios
    results_top_bottom = tester.test_momentum_strategy(
        periods=[6, 3, 1], 
        lag_type='month', 
        method='sum',
        portfolio_method='top_bottom'
    )
    print(f"   Top-bottom portfolios - Final spread: {results_top_bottom['spread_stats'].get('final_spread', 'N/A')}")
    
    # Quintile portfolios
    results_quintiles = tester.test_momentum_strategy(
        periods=[6, 3, 1], 
        lag_type='month', 
        method='sum',
        portfolio_method='quintiles'
    )
    print(f"   Quintile portfolios - Final spread: {results_quintiles['spread_stats'].get('final_spread', 'N/A')}")
    
    # Decile portfolios
    results_deciles = tester.test_momentum_strategy(
        periods=[6, 3, 1], 
        lag_type='month', 
        method='sum',
        portfolio_method='deciles'
    )
    print(f"   Decile portfolios - Final spread: {results_deciles['spread_stats'].get('final_spread', 'N/A')}")
    
    return {
        'simple': [results_6m, results_12m],
        'combined': [results_6_3_1_sum, results_6_3_1_min2, results_6_3_1_mean],
        'lags': [results_daily, results_weekly, results_monthly],
        'portfolios': [results_top_bottom, results_quintiles, results_deciles]
    }

def analyze_momentum_characteristics(tester, periods_list, lag_types):
    """
    Analyze momentum characteristics across different periods and lag types.
    """
    print("\n=== Momentum Characteristics Analysis ===")
    
    results = {}
    
    for lag_type in lag_types:
        print(f"\n{lag_type.upper()} Momentum Characteristics:")
        results[lag_type] = {}
        
        for periods in periods_list:
            momentum_df = tester.calculate_combined_momentum(periods, lag_type, 'sum')
            stats = tester.calculate_momentum_statistics('combined_momentum')
            
            results[lag_type][str(periods)] = stats
            
            print(f"  Periods {periods}:")
            print(f"    Mean: {stats['mean']:.4f}")
            print(f"    Std: {stats['std']:.4f}")
            print(f"    Skewness: {stats['skewness']:.4f}")
            print(f"    Positive ratio: {stats['positive_ratio']:.4f}")
            print(f"    Autocorrelation: {stats['autocorrelation']:.4f}")
    
    return results

def create_momentum_comparison_plot(results_dict):
    """
    Create comparison plots for different momentum strategies.
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: Simple momentum comparison
    simple_results = results_dict['simple']
    periods = ['6m', '12m']
    means = [r['momentum_stats']['mean'] for r in simple_results]
    stds = [r['momentum_stats']['std'] for r in simple_results]
    
    axes[0, 0].bar(periods, means, yerr=stds, capsize=5)
    axes[0, 0].set_title('Simple Momentum Strategies')
    axes[0, 0].set_ylabel('Mean Momentum')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Combined momentum comparison
    combined_results = results_dict['combined']
    labels = ['6+3+1 (sum)', '6+3+1 (min2)', '6+3+1 (mean)']
    means = [r['momentum_stats']['mean'] for r in combined_results]
    positive_ratios = [r['momentum_stats']['positive_ratio'] for r in combined_results]
    
    x = np.arange(len(labels))
    width = 0.35
    
    axes[0, 1].bar(x - width/2, means, width, label='Mean')
    axes[0, 1].bar(x + width/2, positive_ratios, width, label='Positive Ratio')
    axes[0, 1].set_title('Combined Momentum Strategies')
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(labels, rotation=45)
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Different lag types
    lag_results = results_dict['lags']
    lag_labels = ['Daily', 'Weekly', 'Monthly']
    means = [r['momentum_stats']['mean'] for r in lag_results]
    stds = [r['momentum_stats']['std'] for r in lag_results]
    
    axes[1, 0].bar(lag_labels, means, yerr=stds, capsize=5)
    axes[1, 0].set_title('Momentum by Lag Type')
    axes[1, 0].set_ylabel('Mean Momentum')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Portfolio performance
    portfolio_results = results_dict['portfolios']
    portfolio_labels = ['Top-Bottom', 'Quintiles', 'Deciles']
    
    # Extract final spreads if available
    spreads = []
    for result in portfolio_results:
        spread = result['spread_stats'].get('final_spread', 0)
        spreads.append(spread if isinstance(spread, (int, float)) else 0)
    
    axes[1, 1].bar(portfolio_labels, spreads)
    axes[1, 1].set_title('Portfolio Performance (Winner-Loser Spread)')
    axes[1, 1].set_ylabel('Final Spread')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('momentum_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

def demonstrate_flexible_usage():
    """
    Demonstrate flexible usage of the momentum framework.
    """
    print("\n=== Flexible Usage Examples ===")
    
    # Create sample data
    sample_data = create_sample_data(n_days=500, n_assets=50)
    tester = MomentumAnomalyTester(sample_data, price_col='close', date_col='date')
    
    # Example 1: Custom momentum calculation
    print("\n1. Custom Momentum Calculation:")
    momentum_df = tester.calculate_momentum_lags([1, 3, 6, 12], 'month')
    print(f"   Calculated momentum for periods: {[1, 3, 6, 12]} months")
    print(f"   Available momentum columns: {[col for col in momentum_df.columns if 'momentum_' in col]}")
    
    # Example 2: Different combination methods
    print("\n2. Different Combination Methods:")
    periods = [6, 3, 1]
    
    for method in ['sum', 'mean', 'product']:
        combined_df = tester.calculate_combined_momentum(periods, 'month', method)
        stats = tester.calculate_momentum_statistics('combined_momentum')
        print(f"   {method.capitalize()} method - Mean: {stats['mean']:.4f}, Std: {stats['std']:.4f}")
    
    # Example 3: Minimum positive requirement
    print("\n3. Minimum Positive Requirements:")
    for min_positive in [1, 2, 3]:
        combined_df = tester.calculate_combined_momentum(periods, 'month', 'sum', min_positive)
        stats = tester.calculate_momentum_statistics('combined_momentum')
        print(f"   Min {min_positive} positive - Mean: {stats['mean']:.4f}, "
              f"Positive ratio: {stats['positive_ratio']:.4f}")
    
    # Example 4: Portfolio creation
    print("\n4. Portfolio Creation:")
    momentum_df = tester.calculate_combined_momentum([6, 3, 1], 'month', 'sum')
    
    for method in ['top_bottom', 'quintiles', 'deciles']:
        portfolio_df = tester.create_momentum_portfolio('combined_momentum', method=method)
        portfolio_counts = portfolio_df['portfolio'].value_counts()
        print(f"   {method.capitalize()} portfolios: {dict(portfolio_counts)}")

if __name__ == "__main__":
    # Run comprehensive examples
    results = run_momentum_examples()
    
    # Analyze momentum characteristics
    periods_list = [[6], [12], [6, 3, 1], [12, 6, 3]]
    lag_types = ['day', 'week', 'month']
    
    # Create sample data for analysis
    sample_data = create_sample_data(n_days=500, n_assets=50)
    tester = MomentumAnomalyTester(sample_data, price_col='close', date_col='date')
    
    characteristics = analyze_momentum_characteristics(tester, periods_list, lag_types)
    
    # Create comparison plots
    create_momentum_comparison_plot(results)
    
    # Demonstrate flexible usage
    demonstrate_flexible_usage()
    
    print("\n=== Framework Usage Summary ===")
    print("The momentum framework provides:")
    print("✓ Flexible lag periods (day, week, month)")
    print("✓ Multiple period combinations (e.g., 6+3+1 months)")
    print("✓ Different combination methods (sum, mean, product)")
    print("✓ Minimum positive period requirements")
    print("✓ Various portfolio construction methods")
    print("✓ Comprehensive statistical analysis")
    print("✓ No moving averages approach")
    
    print("\nFramework ready for momentum anomaly research!")