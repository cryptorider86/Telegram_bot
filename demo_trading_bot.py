#!/usr/bin/env python3
"""
Демонстрация торгового бота с симулированными данными
Показывает работу бота в реальном времени без внешних API
"""

import time
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from trading_strategy import MovingAverageStrategy, RSIStrategy
from crypto_trading_bot import PaperTradingBot

class SimulatedExchange:
    """Симулированная биржа для демонстрации"""
    
    def __init__(self):
        self.current_price = 50000.0  # Начальная цена BTC
        self.price_history = []
        self.time_step = 0
        
        # Генерируем базовые исторические данные
        np.random.seed(42)
        self._generate_base_data()
    
    def _generate_base_data(self):
        """Генерируем базовые исторические данные"""
        dates = pd.date_range(start=datetime.now() - timedelta(hours=100), 
                             end=datetime.now(), freq='H')
        
        prices = []
        price = self.current_price
        
        for _ in range(len(dates)):
            # Симулируем движение цены с трендом и волатильностью
            change = np.random.normal(0, 100) + np.random.choice([-1, 1]) * 20
            price += change
            price = max(price, 10000)  # Минимальная цена
            prices.append(price)
        
        self.base_data = pd.DataFrame({
            'open': prices,
            'high': [p * (1 + np.random.uniform(0, 0.02)) for p in prices],
            'low': [p * (1 - np.random.uniform(0, 0.02)) for p in prices],
            'close': prices,
            'volume': np.random.randint(1000, 10000, len(dates))
        }, index=dates)
        
        self.current_price = prices[-1]
    
    def get_market_data(self, symbol: str, interval: str = "1h", limit: int = 100) -> pd.DataFrame:
        """Получение исторических данных"""
        return self.base_data.tail(limit).copy()
    
    def get_current_price(self, symbol: str) -> float:
        """Симулируем изменение цены"""
        # Добавляем случайное изменение цены
        change = np.random.normal(0, 50)
        self.current_price += change
        self.current_price = max(self.current_price, 10000)
        
        # Добавляем новую запись в историю
        new_timestamp = datetime.now()
        new_row = pd.DataFrame({
            'open': [self.current_price],
            'high': [self.current_price * (1 + np.random.uniform(0, 0.01))],
            'low': [self.current_price * (1 - np.random.uniform(0, 0.01))],
            'close': [self.current_price],
            'volume': [np.random.randint(1000, 10000)]
        }, index=[new_timestamp])
        
        self.base_data = pd.concat([self.base_data, new_row]).tail(200)
        
        return self.current_price

class DemoTradingBot(PaperTradingBot):
    """Демонстрационный торговый бот с улучшенной статистикой"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.trade_count = 0
        self.max_balance = self.initial_balance
        self.min_balance = self.initial_balance
        
    def start_demo(self, iterations: int = 20, interval: int = 2):
        """Запуск демонстрации"""
        print(f"🤖 Запуск демо торгового бота для {self.symbol}")
        print(f"💰 Начальный баланс: ${self.initial_balance:.2f}")
        print(f"🔄 Количество итераций: {iterations}")
        print(f"⏱️ Интервал: {interval} секунд")
        print("=" * 60)
        
        for i in range(iterations):
            print(f"\n🔍 Итерация {i+1}/{iterations}")
            self.check_signals()
            
            # Обновляем статистику
            status = self.get_portfolio_status()
            self.max_balance = max(self.max_balance, status['total_value'])
            self.min_balance = min(self.min_balance, status['total_value'])
            
            # Показываем текущий статус каждые 5 итераций
            if (i + 1) % 5 == 0:
                self.show_detailed_status()
            
            time.sleep(interval)
        
        print("\n" + "=" * 60)
        print("📊 ФИНАЛЬНЫЕ РЕЗУЛЬТАТЫ")
        print("=" * 60)
        self.show_final_results()
    
    def show_detailed_status(self):
        """Показать детальный статус"""
        status = self.get_portfolio_status()
        print(f"\n💼 ТЕКУЩИЙ СТАТУС:")
        print(f"   💰 Баланс: ${status['balance']:.2f}")
        print(f"   📊 Позиция: {status['position_size']:.6f} BTC")
        print(f"   💵 Общая стоимость: ${status['total_value']:.2f}")
        print(f"   📈 Доходность: {status['total_return']:.2f}%")
        if status['unrealized_pnl'] != 0:
            print(f"   📋 Нереализованная П/У: ${status['unrealized_pnl']:.2f}")
    
    def show_final_results(self):
        """Показать финальные результаты"""
        status = self.get_portfolio_status()
        
        print(f"💰 Начальный баланс: ${self.initial_balance:.2f}")
        print(f"💵 Финальная стоимость: ${status['total_value']:.2f}")
        print(f"📈 Общая доходность: {status['total_return']:.2f}%")
        print(f"📊 Максимальный баланс: ${self.max_balance:.2f}")
        print(f"📉 Минимальный баланс: ${self.min_balance:.2f}")
        print(f"🔄 Всего сделок: {len(self.strategy.trades)}")
        
        if len(self.strategy.trades) > 0:
            buy_trades = [t for t in self.strategy.trades if t.side == 'buy']
            sell_trades = [t for t in self.strategy.trades if t.side == 'sell']
            print(f"📈 Покупок: {len(buy_trades)}")
            print(f"📉 Продаж: {len(sell_trades)}")
        
        # Рассчитываем максимальную просадку
        max_drawdown = ((self.max_balance - self.min_balance) / self.max_balance) * 100
        print(f"⚠️ Максимальная просадка: {max_drawdown:.2f}%")
    
    def buy_signal(self, price: float):
        """Улучшенная обработка сигнала на покупку"""
        if self.position == 0:
            super().buy_signal(price)
            print(f"   🎯 Причина: Сигнал стратегии")
    
    def sell_signal(self, price: float):
        """Улучшенная обработка сигнала на продажу"""
        if self.position == 1:
            super().sell_signal(price)
            print(f"   🎯 Причина: Сигнал стратегии")

def demo_multiple_strategies():
    """Демонстрация нескольких стратегий одновременно"""
    print("🚀 СРАВНЕНИЕ ТОРГОВЫХ СТРАТЕГИЙ В РЕАЛЬНОМ ВРЕМЕНИ")
    print("=" * 70)
    
    exchange = SimulatedExchange()
    
    strategies = [
        ("Moving Average (5/15)", MovingAverageStrategy(fast_period=5, slow_period=15, initial_capital=10000)),
        ("RSI (25/75)", RSIStrategy(oversold=25, overbought=75, initial_capital=10000)),
    ]
    
    bots = []
    for name, strategy in strategies:
        bot = DemoTradingBot(strategy, exchange, "BTC/USDT", initial_balance=10000)
        bots.append((name, bot))
    
    print(f"📊 Начальная цена BTC: ${exchange.current_price:.2f}")
    print("🔄 Запуск симуляции...")
    
    for i in range(15):  # 15 итераций
        print(f"\n⏰ Момент времени {i+1}/15")
        current_price = exchange.get_current_price("BTC/USDT")
        print(f"💰 Текущая цена BTC: ${current_price:.2f}")
        
        for name, bot in bots:
            print(f"\n📈 {name}:")
            bot.check_signals()
            
        time.sleep(1)  # Пауза между итерациями
    
    # Финальные результаты
    print("\n" + "=" * 70)
    print("🏆 ФИНАЛЬНЫЕ РЕЗУЛЬТАТЫ СРАВНЕНИЯ")
    print("=" * 70)
    
    results = []
    for name, bot in bots:
        status = bot.get_portfolio_status()
        results.append((name, status['total_return'], status['total_value']))
        
        print(f"\n📊 {name}:")
        print(f"   💵 Финальная стоимость: ${status['total_value']:.2f}")
        print(f"   📈 Доходность: {status['total_return']:.2f}%")
        print(f"   🔄 Сделок: {len(bot.strategy.trades)}")
    
    # Определяем лучшую стратегию
    best_strategy = max(results, key=lambda x: x[1])
    print(f"\n🏆 ЛУЧШАЯ СТРАТЕГИЯ: {best_strategy[0]}")
    print(f"🎯 Доходность: {best_strategy[1]:.2f}%")
    print(f"💵 Финальная стоимость: ${best_strategy[2]:.2f}")

def main():
    """Главная функция демонстрации"""
    print("🎮 ИНТЕРАКТИВНАЯ ДЕМОНСТРАЦИЯ ТОРГОВОГО БОТА")
    print("=" * 60)
    
    # Демонстрация одной стратегии
    exchange = SimulatedExchange()
    strategy = MovingAverageStrategy(fast_period=8, slow_period=20, initial_capital=10000)
    bot = DemoTradingBot(strategy, exchange, "BTC/USDT", initial_balance=10000)
    
    print("🎯 Демонстрация Moving Average стратегии")
    bot.start_demo(iterations=15, interval=1)
    
    print("\n" + "🔄" * 30)
    time.sleep(2)
    
    # Сравнение стратегий
    demo_multiple_strategies()
    
    print("\n✅ Демонстрация завершена!")
    print("📝 Для реальной торговли:")
    print("   1. Получите API ключи с биржи")
    print("   2. Используйте LiveTradingBot")
    print("   3. Начните с малых сумм")
    print("   4. Установите стоп-лоссы")

if __name__ == "__main__":
    main()