#!/usr/bin/env python3
"""
Торговая стратегия с техническими индикаторами
Поддерживает различные стратегии и бэктестинг
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
import requests
import time

@dataclass
class Trade:
    """Класс для хранения информации о сделке"""
    timestamp: datetime
    symbol: str
    side: str  # 'buy' или 'sell'
    price: float
    quantity: float
    commission: float = 0.001  # 0.1% комиссия по умолчанию

class TechnicalIndicators:
    """Класс для расчета технических индикаторов"""
    
    @staticmethod
    def sma(data: pd.Series, period: int) -> pd.Series:
        """Простая скользящая средняя"""
        return data.rolling(window=period).mean()
    
    @staticmethod
    def ema(data: pd.Series, period: int) -> pd.Series:
        """Экспоненциальная скользящая средняя"""
        return data.ewm(span=period).mean()
    
    @staticmethod
    def rsi(data: pd.Series, period: int = 14) -> pd.Series:
        """Индекс относительной силы"""
        delta = data.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    @staticmethod
    def bollinger_bands(data: pd.Series, period: int = 20, std_dev: float = 2) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Полосы Боллинджера"""
        sma = data.rolling(window=period).mean()
        std = data.rolling(window=period).std()
        upper_band = sma + (std * std_dev)
        lower_band = sma - (std * std_dev)
        return upper_band, sma, lower_band
    
    @staticmethod
    def macd(data: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """MACD индикатор"""
        ema_fast = data.ewm(span=fast).mean()
        ema_slow = data.ewm(span=slow).mean()
        macd = ema_fast - ema_slow
        signal_line = macd.ewm(span=signal).mean()
        histogram = macd - signal_line
        return macd, signal_line, histogram

class TradingStrategy:
    """Базовый класс для торговых стратегий"""
    
    def __init__(self, initial_capital: float = 10000, commission: float = 0.001):
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.commission = commission
        self.position = 0  # Текущая позиция
        self.trades: List[Trade] = []
        self.equity_curve: List[float] = [initial_capital]
        
    def calculate_position_size(self, price: float, risk_percent: float = 0.02) -> float:
        """Расчет размера позиции на основе риск-менеджмента"""
        risk_amount = self.current_capital * risk_percent
        return risk_amount / price
    
    def execute_trade(self, timestamp: datetime, symbol: str, side: str, price: float, quantity: float):
        """Выполнение сделки"""
        commission_cost = price * quantity * self.commission
        
        if side == 'buy':
            cost = price * quantity + commission_cost
            if cost <= self.current_capital:
                self.current_capital -= cost
                self.position += quantity
                trade = Trade(timestamp, symbol, side, price, quantity, commission_cost)
                self.trades.append(trade)
        elif side == 'sell':
            if quantity <= self.position:
                revenue = price * quantity - commission_cost
                self.current_capital += revenue
                self.position -= quantity
                trade = Trade(timestamp, symbol, side, price, quantity, commission_cost)
                self.trades.append(trade)
        
        # Обновляем кривую капитала
        total_value = self.current_capital + (self.position * price)
        self.equity_curve.append(total_value)
    
    def get_performance_stats(self) -> Dict:
        """Получение статистики производительности"""
        if len(self.equity_curve) < 2:
            return {}
        
        final_value = self.equity_curve[-1]
        total_return = (final_value - self.initial_capital) / self.initial_capital
        
        # Расчет максимальной просадки
        peak = self.initial_capital
        max_drawdown = 0
        for value in self.equity_curve:
            if value > peak:
                peak = value
            drawdown = (peak - value) / peak
            max_drawdown = max(max_drawdown, drawdown)
        
        # Расчет Sharpe ratio (упрощенный)
        returns = pd.Series(self.equity_curve).pct_change().dropna()
        sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0
        
        return {
            'total_return': total_return,
            'final_value': final_value,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'total_trades': len(self.trades),
            'win_rate': self.calculate_win_rate()
        }
    
    def calculate_win_rate(self) -> float:
        """Расчет процента прибыльных сделок"""
        if len(self.trades) < 2:
            return 0
        
        profitable_trades = 0
        for i in range(0, len(self.trades), 2):
            if i + 1 < len(self.trades):
                buy_trade = self.trades[i]
                sell_trade = self.trades[i + 1]
                if sell_trade.price > buy_trade.price:
                    profitable_trades += 1
        
        return profitable_trades / (len(self.trades) // 2) if len(self.trades) > 1 else 0

class MovingAverageStrategy(TradingStrategy):
    """Стратегия на основе скользящих средних"""
    
    def __init__(self, fast_period: int = 10, slow_period: int = 30, **kwargs):
        super().__init__(**kwargs)
        self.fast_period = fast_period
        self.slow_period = slow_period
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Генерация торговых сигналов"""
        df = data.copy()
        
        # Расчет скользящих средних
        df['sma_fast'] = TechnicalIndicators.sma(df['close'], self.fast_period)
        df['sma_slow'] = TechnicalIndicators.sma(df['close'], self.slow_period)
        
        # Генерация сигналов
        df['signal'] = 0
        df['signal'][self.fast_period:] = np.where(
            df['sma_fast'][self.fast_period:] > df['sma_slow'][self.fast_period:], 1, 0
        )
        df['position'] = df['signal'].diff()
        
        return df

class RSIStrategy(TradingStrategy):
    """Стратегия на основе RSI"""
    
    def __init__(self, rsi_period: int = 14, oversold: float = 30, overbought: float = 70, **kwargs):
        super().__init__(**kwargs)
        self.rsi_period = rsi_period
        self.oversold = oversold
        self.overbought = overbought
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Генерация торговых сигналов на основе RSI"""
        df = data.copy()
        
        # Расчет RSI
        df['rsi'] = TechnicalIndicators.rsi(df['close'], self.rsi_period)
        
        # Генерация сигналов
        df['signal'] = 0
        df.loc[df['rsi'] < self.oversold, 'signal'] = 1  # Покупка
        df.loc[df['rsi'] > self.overbought, 'signal'] = -1  # Продажа
        
        return df

class BollingerBandsStrategy(TradingStrategy):
    """Стратегия на основе полос Боллинджера"""
    
    def __init__(self, bb_period: int = 20, bb_std: float = 2, **kwargs):
        super().__init__(**kwargs)
        self.bb_period = bb_period
        self.bb_std = bb_std
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Генерация торговых сигналов на основе полос Боллинджера"""
        df = data.copy()
        
        # Расчет полос Боллинджера
        df['bb_upper'], df['bb_middle'], df['bb_lower'] = TechnicalIndicators.bollinger_bands(
            df['close'], self.bb_period, self.bb_std
        )
        
        # Генерация сигналов
        df['signal'] = 0
        df.loc[df['close'] < df['bb_lower'], 'signal'] = 1  # Покупка при касании нижней полосы
        df.loc[df['close'] > df['bb_upper'], 'signal'] = -1  # Продажа при касании верхней полосы
        
        return df

class Backtester:
    """Класс для бэктестинга торговых стратегий"""
    
    def __init__(self, strategy: TradingStrategy):
        self.strategy = strategy
    
    def run_backtest(self, data: pd.DataFrame, symbol: str = 'BTC/USD') -> Dict:
        """Запуск бэктестинга"""
        df = self.strategy.generate_signals(data)
        
        position = 0
        for i, row in df.iterrows():
            timestamp = row.name if isinstance(row.name, datetime) else datetime.now()
            price = row['close']
            
            if hasattr(df, 'position') and not pd.isna(row.get('position', 0)):
                if row['position'] == 1 and position == 0:  # Покупка
                    quantity = self.strategy.calculate_position_size(price)
                    self.strategy.execute_trade(timestamp, symbol, 'buy', price, quantity)
                    position = 1
                elif row['position'] == -1 and position == 1:  # Продажа
                    self.strategy.execute_trade(timestamp, symbol, 'sell', price, self.strategy.position)
                    position = 0
            
            elif 'signal' in df.columns and not pd.isna(row.get('signal', 0)):
                if row['signal'] == 1 and position == 0:  # Покупка
                    quantity = self.strategy.calculate_position_size(price)
                    self.strategy.execute_trade(timestamp, symbol, 'buy', price, quantity)
                    position = 1
                elif row['signal'] == -1 and position == 1:  # Продажа
                    self.strategy.execute_trade(timestamp, symbol, 'sell', price, self.strategy.position)
                    position = 0
        
        return self.strategy.get_performance_stats()
    
    def plot_results(self, data: pd.DataFrame):
        """Построение графиков результатов"""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # График цены и сигналов
        ax1.plot(data.index, data['close'], label='Цена', alpha=0.7)
        
        if hasattr(self.strategy, 'fast_period'):  # Moving Average Strategy
            df = self.strategy.generate_signals(data)
            ax1.plot(data.index, df['sma_fast'], label=f'SMA {self.strategy.fast_period}', alpha=0.7)
            ax1.plot(data.index, df['sma_slow'], label=f'SMA {self.strategy.slow_period}', alpha=0.7)
        
        ax1.set_title('Цена и торговые сигналы')
        ax1.legend()
        ax1.grid(True)
        
        # График кривой капитала
        ax2.plot(self.strategy.equity_curve, label='Кривая капитала')
        ax2.axhline(y=self.strategy.initial_capital, color='r', linestyle='--', label='Начальный капитал')
        ax2.set_title('Кривая капитала')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.show()

def get_sample_data() -> pd.DataFrame:
    """Получение примерных данных для тестирования"""
    # Создаем синтетические данные для демонстрации
    np.random.seed(42)
    dates = pd.date_range(start='2023-01-01', end='2024-01-01', freq='D')
    
    # Генерируем случайные цены с трендом
    price = 100
    prices = []
    for _ in range(len(dates)):
        price += np.random.normal(0, 2) + 0.01  # Небольшой восходящий тренд
        prices.append(max(price, 10))  # Минимальная цена 10
    
    df = pd.DataFrame({
        'open': prices,
        'high': [p * (1 + np.random.uniform(0, 0.05)) for p in prices],
        'low': [p * (1 - np.random.uniform(0, 0.05)) for p in prices],
        'close': prices,
        'volume': np.random.randint(1000, 10000, len(dates))
    }, index=dates)
    
    return df

def main():
    """Основная функция для демонстрации"""
    print("🚀 Демонстрация торговых стратегий")
    print("=" * 50)
    
    # Получаем данные
    data = get_sample_data()
    print(f"📊 Загружено {len(data)} дней данных")
    
    # Тестируем разные стратегии
    strategies = [
        ("Moving Average (10/30)", MovingAverageStrategy(fast_period=10, slow_period=30)),
        ("RSI (30/70)", RSIStrategy(oversold=30, overbought=70)),
        ("Bollinger Bands", BollingerBandsStrategy())
    ]
    
    results = {}
    
    for name, strategy in strategies:
        print(f"\n📈 Тестирование стратегии: {name}")
        backtester = Backtester(strategy)
        performance = backtester.run_backtest(data)
        results[name] = performance
        
        print(f"💰 Итоговая доходность: {performance.get('total_return', 0):.2%}")
        print(f"💵 Финальная стоимость: ${performance.get('final_value', 0):.2f}")
        print(f"📉 Максимальная просадка: {performance.get('max_drawdown', 0):.2%}")
        print(f"📊 Коэффициент Шарпа: {performance.get('sharpe_ratio', 0):.2f}")
        print(f"🎯 Процент прибыльных сделок: {performance.get('win_rate', 0):.2%}")
        print(f"🔄 Всего сделок: {performance.get('total_trades', 0)}")
    
    # Сравнение результатов
    print("\n" + "=" * 50)
    print("📊 СРАВНЕНИЕ СТРАТЕГИЙ")
    print("=" * 50)
    
    best_strategy = max(results.items(), key=lambda x: x[1].get('total_return', 0))
    print(f"🏆 Лучшая стратегия: {best_strategy[0]}")
    print(f"🎯 Доходность: {best_strategy[1].get('total_return', 0):.2%}")

if __name__ == "__main__":
    main()