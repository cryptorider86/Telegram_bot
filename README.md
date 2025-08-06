# Momentum Anomaly Testing Framework

A flexible and comprehensive framework for testing momentum anomalies with different lags, period combinations, and portfolio construction methods. This framework is designed for academic research and practical trading strategy development.

## Features

- **Flexible Lag Periods**: Test momentum with daily, weekly, or monthly lags
- **Multiple Period Combinations**: Combine different periods (e.g., 6+3+1 months)
- **No Moving Averages**: Pure momentum approach without smoothing
- **Minimum Positive Requirements**: Require specific number of positive periods
- **Multiple Portfolio Methods**: Top-bottom, quintiles, deciles
- **Comprehensive Statistics**: Mean, std, skewness, kurtosis, autocorrelation
- **Visualization**: Built-in plotting capabilities

## Installation

1. Clone or download the framework files
2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Quick Start

```python
from momentum_anomaly_framework import MomentumAnomalyTester, create_sample_data

# Create sample data
data = create_sample_data(n_days=1000, n_assets=100)

# Initialize tester
tester = MomentumAnomalyTester(data, price_col='close', date_col='date')

# Test 6+3+1 month momentum strategy
results = tester.test_momentum_strategy(
    periods=[6, 3, 1], 
    lag_type='month', 
    method='sum',
    min_positive=2  # At least 2 periods must be positive
)

print(f"Momentum mean: {results['momentum_stats']['mean']:.4f}")
print(f"Positive ratio: {results['momentum_stats']['positive_ratio']:.4f}")
```

## Framework Components

### 1. MomentumAnomalyTester Class

The main class for momentum analysis with methods:

- `calculate_momentum_lags()`: Calculate momentum for different lag periods
- `calculate_combined_momentum()`: Combine multiple periods with different methods
- `create_momentum_portfolio()`: Create portfolios based on momentum signals
- `calculate_portfolio_returns()`: Calculate portfolio performance
- `calculate_momentum_statistics()`: Compute comprehensive statistics
- `test_momentum_strategy()`: Complete strategy testing

### 2. Key Parameters

#### Lag Types
- `'day'`: Daily momentum (1 day = 1 trading day)
- `'week'`: Weekly momentum (1 week = 5 trading days)
- `'month'`: Monthly momentum (1 month = 21 trading days)

#### Combination Methods
- `'sum'`: Sum of individual momentum measures
- `'mean'`: Average of individual momentum measures
- `'product'`: Product of individual momentum measures

#### Portfolio Methods
- `'top_bottom'`: Top 10% and bottom 10% performers
- `'quintiles'`: Five equal-sized portfolios
- `'deciles'`: Ten equal-sized portfolios

## Usage Examples

### Example 1: Simple Momentum Strategy

```python
# Test 6-month momentum
results = tester.test_momentum_strategy(periods=[6], lag_type='month')
print(f"6-month momentum mean: {results['momentum_stats']['mean']:.4f}")
```

### Example 2: Combined Momentum Strategy

```python
# Test 6+3+1 month strategy with minimum 2 positive periods
results = tester.test_momentum_strategy(
    periods=[6, 3, 1], 
    lag_type='month', 
    method='sum',
    min_positive=2
)
```

### Example 3: Different Lag Types

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
```

### Example 4: Portfolio Construction

```python
# Calculate momentum
momentum_df = tester.calculate_combined_momentum([6, 3, 1], 'month', 'sum')

# Create portfolios
portfolio_df = tester.create_momentum_portfolio('combined_momentum', method='quintiles')

# Calculate returns
returns = tester.calculate_portfolio_returns(portfolio_df)
```

## Advanced Usage

### Custom Momentum Calculation

```python
# Calculate momentum for specific periods
momentum_df = tester.calculate_momentum_lags([1, 3, 6, 12], 'month')

# Access individual momentum measures
momentum_1m = momentum_df['momentum_1m']
momentum_3m = momentum_df['momentum_3m']
momentum_6m = momentum_df['momentum_6m']
momentum_12m = momentum_df['momentum_12m']
```

### Statistical Analysis

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

### Multiple Strategy Comparison

```python
# Test different strategies
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

## Data Requirements

Your data should have the following columns:
- **Date column**: Datetime format
- **Price column**: Numeric price data
- **Asset ID**: Optional, for multi-asset analysis

Example data structure:
```python
data = pd.DataFrame({
    'date': ['2020-01-01', '2020-01-02', ...],
    'close': [100.0, 101.5, ...],
    'asset_id': [1, 1, ...]  # Optional
})
```

## Running Examples

To run the comprehensive examples:

```bash
python momentum_examples.py
```

This will:
1. Test various momentum strategies
2. Analyze momentum characteristics
3. Create comparison plots
4. Demonstrate flexible usage

## Output Files

- `momentum_comparison.png`: Comparison plots of different strategies
- Console output with detailed statistics and analysis

## Research Applications

This framework is particularly useful for:

1. **Academic Research**: Testing momentum anomalies across different markets
2. **Strategy Development**: Developing and backtesting momentum strategies
3. **Risk Management**: Analyzing momentum characteristics and risks
4. **Market Analysis**: Understanding momentum patterns in different timeframes

## Key Advantages

- **No Moving Averages**: Pure momentum approach without smoothing
- **Flexible Periods**: Test any combination of periods
- **Multiple Methods**: Sum, mean, or product combinations
- **Minimum Requirements**: Require specific number of positive periods
- **Comprehensive Stats**: Full statistical analysis
- **Easy Extension**: Modular design for custom modifications

## Contributing

Feel free to extend the framework with:
- Additional portfolio construction methods
- More statistical measures
- Custom momentum calculations
- Additional visualization options

## License

This framework is provided for educational and research purposes.
