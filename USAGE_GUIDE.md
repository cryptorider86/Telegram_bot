# Momentum Anomaly Testing Framework - Usage Guide

## Overview

This framework provides a flexible and comprehensive approach to testing momentum anomalies with different lags, period combinations, and portfolio construction methods. It's designed specifically for academic research and practical trading strategy development.

## Key Features

### 1. Flexible Lag Periods
- **Daily**: Test momentum with daily lags (1 day = 1 trading day)
- **Weekly**: Test momentum with weekly lags (1 week = 5 trading days)
- **Monthly**: Test momentum with monthly lags (1 month = 21 trading days)

### 2. Multiple Period Combinations
- Combine different periods (e.g., 6+3+1 months)
- Test any combination of periods you want
- No moving averages approach - pure momentum

### 3. Minimum Positive Requirements
- Require specific number of positive periods
- Example: 6+3+1 with min_positive=2 means at least 2 periods must be positive

### 4. Multiple Combination Methods
- **Sum**: Add individual momentum measures
- **Mean**: Average individual momentum measures
- **Product**: Multiply individual momentum measures

### 5. Portfolio Construction Methods
- **Top-Bottom**: Top 10% and bottom 10% performers
- **Quintiles**: Five equal-sized portfolios
- **Deciles**: Ten equal-sized portfolios

## Installation

```bash
pip install pandas numpy matplotlib seaborn
```

## Basic Usage

### Step 1: Import the Framework

```python
from momentum_anomaly_framework import MomentumAnomalyTester, create_sample_data
```

### Step 2: Prepare Your Data

Your data should have this structure:
```python
import pandas as pd

# Example data structure
data = pd.DataFrame({
    'date': ['2020-01-01', '2020-01-02', '2020-01-03', ...],
    'close': [100.0, 101.5, 99.8, ...],
    'asset_id': [1, 1, 1, ...]  # Optional
})
```

### Step 3: Initialize the Tester

```python
tester = MomentumAnomalyTester(data, price_col='close', date_col='date')
```

### Step 4: Test Momentum Strategies

```python
# Simple 6-month momentum
results = tester.test_momentum_strategy(periods=[6], lag_type='month')

# Combined 6+3+1 month momentum
results = tester.test_momentum_strategy(
    periods=[6, 3, 1], 
    lag_type='month', 
    method='sum'
)

# With minimum positive requirement
results = tester.test_momentum_strategy(
    periods=[6, 3, 1], 
    lag_type='month', 
    method='sum',
    min_positive=2  # At least 2 periods must be positive
)
```

## Advanced Usage Examples

### Example 1: Different Lag Types

```python
# Daily momentum
daily_results = tester.test_momentum_strategy(
    periods=[5, 10, 20], 
    lag_type='day', 
    method='sum'
)

# Weekly momentum
weekly_results = tester.test_momentum_strategy(
    periods=[4, 8, 12], 
    lag_type='week', 
    method='sum'
)

# Monthly momentum
monthly_results = tester.test_momentum_strategy(
    periods=[3, 6, 12], 
    lag_type='month', 
    method='sum'
)
```

### Example 2: Different Combination Methods

```python
periods = [6, 3, 1]

# Sum method
sum_results = tester.test_momentum_strategy(
    periods=periods, 
    lag_type='month', 
    method='sum'
)

# Mean method
mean_results = tester.test_momentum_strategy(
    periods=periods, 
    lag_type='month', 
    method='mean'
)

# Product method
product_results = tester.test_momentum_strategy(
    periods=periods, 
    lag_type='month', 
    method='product'
)
```

### Example 3: Portfolio Construction

```python
# Calculate momentum
momentum_df = tester.calculate_combined_momentum([6, 3, 1], 'month', 'sum')

# Create different portfolio types
top_bottom_portfolio = tester.create_momentum_portfolio(
    'combined_momentum', 
    method='top_bottom'
)

quintile_portfolio = tester.create_momentum_portfolio(
    'combined_momentum', 
    method='quintiles'
)

decile_portfolio = tester.create_momentum_portfolio(
    'combined_momentum', 
    method='deciles'
)
```

### Example 4: Statistical Analysis

```python
# Get comprehensive statistics
stats = tester.calculate_momentum_statistics('combined_momentum')

print(f"Mean: {stats['mean']:.4f}")
print(f"Standard Deviation: {stats['std']:.4f}")
print(f"Skewness: {stats['skewness']:.4f}")
print(f"Kurtosis: {stats['kurtosis']:.4f}")
print(f"Autocorrelation: {stats['autocorrelation']:.4f}")
print(f"Positive Ratio: {stats['positive_ratio']:.4f}")
```

## Research Applications

### 1. Academic Research
```python
# Test momentum across different markets
strategies = [
    {'periods': [6], 'lag_type': 'month'},
    {'periods': [12], 'lag_type': 'month'},
    {'periods': [6, 3, 1], 'lag_type': 'month', 'method': 'sum'},
    {'periods': [6, 3, 1], 'lag_type': 'month', 'method': 'sum', 'min_positive': 2}
]

results = {}
for i, strategy in enumerate(strategies):
    results[f'strategy_{i+1}'] = tester.test_momentum_strategy(**strategy)
```

### 2. Strategy Development
```python
# Test different momentum combinations
momentum_periods = [
    [6], [12], [6, 3, 1], [12, 6, 3], [6, 3, 1, 1]
]

for periods in momentum_periods:
    results = tester.test_momentum_strategy(
        periods=periods, 
        lag_type='month', 
        method='sum'
    )
    print(f"Periods {periods}: Mean = {results['momentum_stats']['mean']:.4f}")
```

### 3. Risk Analysis
```python
# Analyze momentum characteristics
momentum_df = tester.calculate_momentum_lags([1, 3, 6, 12], 'month')

# Calculate statistics for each period
for col in momentum_df.columns:
    if 'momentum_' in col:
        stats = tester.calculate_momentum_statistics(momentum_df[col])
        print(f"{col}: Mean = {stats['mean']:.4f}, Std = {stats['std']:.4f}")
```

## Custom Extensions

### Adding New Combination Methods

```python
# Extend the framework with custom combination methods
def custom_combination_method(momentum_values):
    # Your custom logic here
    return weighted_sum(momentum_values)

# Use in your analysis
momentum_df = tester.calculate_combined_momentum([6, 3, 1], 'month', 'sum')
# Apply custom method to momentum_df
```

### Adding New Portfolio Methods

```python
# Create custom portfolio construction
def custom_portfolio_method(momentum_values, n_portfolios=5):
    # Your custom portfolio logic here
    return portfolio_assignments

# Use in your analysis
momentum_df = tester.calculate_combined_momentum([6, 3, 1], 'month', 'sum')
# Apply custom portfolio method
```

## Best Practices

### 1. Data Preparation
- Ensure your date column is in datetime format
- Sort data by date and asset_id
- Handle missing values appropriately
- Use consistent price data (adjusted close prices recommended)

### 2. Strategy Testing
- Test multiple period combinations
- Compare different lag types
- Use minimum positive requirements to filter signals
- Analyze both individual and combined momentum measures

### 3. Statistical Analysis
- Always calculate comprehensive statistics
- Test for robustness across different time periods
- Consider transaction costs and implementation issues
- Use appropriate significance tests

### 4. Portfolio Construction
- Test different portfolio construction methods
- Consider rebalancing frequency
- Analyze winner-loser spreads
- Monitor portfolio turnover

## Common Research Questions

### 1. What is the optimal momentum period?
```python
periods_to_test = [1, 3, 6, 9, 12, 18, 24]
results = {}

for period in periods_to_test:
    results[period] = tester.test_momentum_strategy(
        periods=[period], 
        lag_type='month'
    )
```

### 2. Does combining multiple periods improve performance?
```python
combinations = [
    [6], [3], [1],
    [6, 3], [6, 1], [3, 1],
    [6, 3, 1]
]

for combo in combinations:
    results = tester.test_momentum_strategy(
        periods=combo, 
        lag_type='month', 
        method='sum'
    )
```

### 3. What is the impact of minimum positive requirements?
```python
for min_positive in [1, 2, 3]:
    results = tester.test_momentum_strategy(
        periods=[6, 3, 1], 
        lag_type='month', 
        method='sum',
        min_positive=min_positive
    )
```

## Output Interpretation

### Strategy Results
```python
results = tester.test_momentum_strategy([6, 3, 1], 'month', 'sum')

# Access results
momentum_stats = results['momentum_stats']
portfolio_returns = results['portfolio_returns']
spread_stats = results['spread_stats']
parameters = results['parameters']
```

### Key Statistics
- **Mean**: Average momentum value
- **Std**: Standard deviation of momentum
- **Skewness**: Distribution asymmetry
- **Kurtosis**: Distribution peakedness
- **Autocorrelation**: Momentum persistence
- **Positive Ratio**: Proportion of positive momentum periods

## Troubleshooting

### Common Issues

1. **Data Format**: Ensure date column is datetime
2. **Missing Values**: Handle NaN values before analysis
3. **Period Length**: Ensure sufficient data for longest period
4. **Memory Usage**: For large datasets, consider chunking

### Performance Tips

1. **Vectorization**: Use pandas operations for speed
2. **Caching**: Store intermediate results for repeated analysis
3. **Parallel Processing**: Use multiprocessing for large datasets
4. **Memory Management**: Clear intermediate variables

## Conclusion

This framework provides a comprehensive and flexible approach to momentum anomaly research. It allows you to:

- Test any combination of momentum periods
- Use different lag types and combination methods
- Apply minimum positive requirements
- Construct various portfolio types
- Perform comprehensive statistical analysis

The framework is designed to be easily extensible for custom research needs while maintaining the rigor required for academic research.