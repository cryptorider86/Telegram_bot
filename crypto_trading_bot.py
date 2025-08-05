#!/usr/bin/env python3
"""
Криптовалютный торговый бот с интеграцией API биржи
Поддерживает реальную торговлю и получение рыночных данных
"""

import time
import json
import hmac
import hashlib
import requests
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from trading_strategy import TradingStrategy, MovingAverageStrategy, RSIStrategy

class CryptoExchangeAPI:
    """Базовый класс для работы с API криптобиржи"""
    
    def __init__(self, api_key: str = "", api_secret: str = "", base_url: str = ""):
        self.api_key = api_key
        self.api_secret = api_secret
        self.base_url = base_url
        self.session = requests.Session()
    
    def get_market_data(self, symbol: str, interval: str = "1h", limit: int = 100) -> pd.DataFrame:
        """Получение рыночных данных"""
        # Это пример, в реальности нужно использовать API конкретной биржи
        # Например, для Binance API
        raise NotImplementedError("Реализуйте для конкретной биржи")
    
    def place_order(self, symbol: str, side: str, quantity: float, price: Optional[float] = None) -> Dict:
        """Размещение ордера"""
        raise NotImplementedError("Реализуйте для конкретной биржи")
    
    def get_account_balance(self) -> Dict:
        """Получение баланса аккаунта"""
        raise NotImplementedError("Реализуйте для конкретной биржи")

class BinanceAPI(CryptoExchangeAPI):
    """API для работы с Binance"""
    
    def __init__(self, api_key: str = "", api_secret: str = ""):
        super().__init__(api_key, api_secret, "https://api.binance.com")
    
    def _generate_signature(self, params: Dict) -> str:
        """Генерация подписи для Binance API"""
        query_string = "&".join([f"{k}={v}" for k, v in params.items()])
        return hmac.new(
            self.api_secret.encode('utf-8'),
            query_string.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
    
    def get_market_data(self, symbol: str, interval: str = "1h", limit: int = 100) -> pd.DataFrame:
        """Получение исторических данных с Binance"""
        endpoint = "/api/v3/klines"
        params = {
            "symbol": symbol.replace("/", ""),  # BTCUSDT вместо BTC/USDT
            "interval": interval,
            "limit": limit
        }
        
        try:
            response = self.session.get(f"{self.base_url}{endpoint}", params=params)
            response.raise_for_status()
            data = response.json()
            
            # Преобразуем в DataFrame
            df = pd.DataFrame(data, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_asset_volume', 'number_of_trades',
                'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
            ])
            
            # Преобразуем типы данных
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df = df.set_index('timestamp')
            
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = pd.to_numeric(df[col])
            
            return df[['open', 'high', 'low', 'close', 'volume']]
            
        except Exception as e:
            print(f"Ошибка получения данных: {e}")
            return pd.DataFrame()
    
    def get_current_price(self, symbol: str) -> float:
        """Получение текущей цены"""
        endpoint = "/api/v3/ticker/price"
        params = {"symbol": symbol.replace("/", "")}
        
        try:
            response = self.session.get(f"{self.base_url}{endpoint}", params=params)
            response.raise_for_status()
            data = response.json()
            return float(data['price'])
        except Exception as e:
            print(f"Ошибка получения цены: {e}")
            return 0.0
    
    def place_order(self, symbol: str, side: str, quantity: float, order_type: str = "MARKET") -> Dict:
        """Размещение ордера (требует API ключи)"""
        if not self.api_key or not self.api_secret:
            return {"error": "API ключи не настроены"}
        
        endpoint = "/api/v3/order"
        timestamp = int(time.time() * 1000)
        
        params = {
            "symbol": symbol.replace("/", ""),
            "side": side.upper(),
            "type": order_type,
            "quantity": quantity,
            "timestamp": timestamp
        }
        
        params["signature"] = self._generate_signature(params)
        
        headers = {"X-MBX-APIKEY": self.api_key}
        
        try:
            response = self.session.post(f"{self.base_url}{endpoint}", data=params, headers=headers)
            return response.json()
        except Exception as e:
            return {"error": str(e)}

class LiveTradingBot:
    """Бот для реальной торговли"""
    
    def __init__(self, strategy: TradingStrategy, exchange: CryptoExchangeAPI, symbol: str = "BTC/USDT"):
        self.strategy = strategy
        self.exchange = exchange
        self.symbol = symbol
        self.is_running = False
        self.position = 0  # 0 = нет позиции, 1 = длинная позиция
        self.last_signal = 0
        
    def start_trading(self, check_interval: int = 300):  # Проверка каждые 5 минут
        """Запуск торгового бота"""
        self.is_running = True
        print(f"🤖 Запуск торгового бота для {self.symbol}")
        print(f"⏰ Интервал проверки: {check_interval} секунд")
        
        while self.is_running:
            try:
                self.check_signals()
                time.sleep(check_interval)
            except KeyboardInterrupt:
                print("\n⏹️ Остановка бота...")
                self.is_running = False
            except Exception as e:
                print(f"❌ Ошибка в боте: {e}")
                time.sleep(60)  # Ждем минуту при ошибке
    
    def check_signals(self):
        """Проверка торговых сигналов"""
        try:
            # Получаем свежие данные
            data = self.exchange.get_market_data(self.symbol, interval="1h", limit=50)
            if data.empty:
                print("⚠️ Не удалось получить данные")
                return
            
            # Генерируем сигналы
            signals_df = self.strategy.generate_signals(data)
            current_signal = signals_df['signal'].iloc[-1] if 'signal' in signals_df.columns else 0
            current_price = data['close'].iloc[-1]
            
            print(f"📊 Текущая цена {self.symbol}: ${current_price:.2f}")
            print(f"📈 Текущий сигнал: {current_signal}")
            
            # Выполняем торговые действия
            if current_signal != self.last_signal:
                if current_signal == 1 and self.position == 0:  # Сигнал на покупку
                    self.buy_signal(current_price)
                elif current_signal == -1 and self.position == 1:  # Сигнал на продажу
                    self.sell_signal(current_price)
                
                self.last_signal = current_signal
            
        except Exception as e:
            print(f"❌ Ошибка при проверке сигналов: {e}")
    
    def buy_signal(self, price: float):
        """Обработка сигнала на покупку"""
        print(f"🟢 СИГНАЛ НА ПОКУПКУ по цене ${price:.2f}")
        
        # В реальной торговле здесь будет вызов API
        # result = self.exchange.place_order(self.symbol, "BUY", quantity)
        
        # Для демонстрации просто логируем
        quantity = self.strategy.calculate_position_size(price)
        print(f"💰 Размер позиции: {quantity:.6f}")
        self.position = 1
        
        # Отправляем уведомление (можно интегрировать с Telegram)
        self.send_notification(f"🟢 Покупка {self.symbol} по ${price:.2f}")
    
    def sell_signal(self, price: float):
        """Обработка сигнала на продажу"""
        print(f"🔴 СИГНАЛ НА ПРОДАЖУ по цене ${price:.2f}")
        
        # В реальной торговле здесь будет вызов API
        # result = self.exchange.place_order(self.symbol, "SELL", self.position_size)
        
        # Для демонстрации просто логируем
        print(f"💸 Продажа позиции")
        self.position = 0
        
        # Отправляем уведомление
        self.send_notification(f"🔴 Продажа {self.symbol} по ${price:.2f}")
    
    def send_notification(self, message: str):
        """Отправка уведомлений (можно интегрировать с Telegram)"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"📱 [{timestamp}] {message}")
        
        # Здесь можно добавить отправку в Telegram
        # telegram_bot.send_message(chat_id, message)
    
    def stop_trading(self):
        """Остановка торгового бота"""
        self.is_running = False
        print("🛑 Торговый бот остановлен")

class PaperTradingBot(LiveTradingBot):
    """Бот для бумажной торговли (без реальных денег)"""
    
    def __init__(self, strategy: TradingStrategy, exchange: CryptoExchangeAPI, 
                 symbol: str = "BTC/USDT", initial_balance: float = 10000):
        super().__init__(strategy, exchange, symbol)
        self.balance = initial_balance
        self.initial_balance = initial_balance
        self.position_size = 0
        self.entry_price = 0
        
    def buy_signal(self, price: float):
        """Бумажная покупка"""
        if self.position == 0:
            position_value = self.balance * 0.95  # Используем 95% баланса
            self.position_size = position_value / price
            self.entry_price = price
            self.balance = self.balance * 0.05  # Оставляем 5% в качестве резерва
            self.position = 1
            
            print(f"🟢 БУМАЖНАЯ ПОКУПКА:")
            print(f"   💰 Цена входа: ${price:.2f}")
            print(f"   📊 Размер позиции: {self.position_size:.6f}")
            print(f"   💵 Остаток баланса: ${self.balance:.2f}")
    
    def sell_signal(self, price: float):
        """Бумажная продажа"""
        if self.position == 1:
            revenue = self.position_size * price
            profit = revenue - (self.position_size * self.entry_price)
            profit_percent = (profit / (self.position_size * self.entry_price)) * 100
            
            self.balance += revenue
            self.position = 0
            
            print(f"🔴 БУМАЖНАЯ ПРОДАЖА:")
            print(f"   💰 Цена выхода: ${price:.2f}")
            print(f"   📊 Прибыль: ${profit:.2f} ({profit_percent:.2f}%)")
            print(f"   💵 Новый баланс: ${self.balance:.2f}")
            
            # Сброс позиции
            self.position_size = 0
            self.entry_price = 0
    
    def get_portfolio_status(self) -> Dict:
        """Получение статуса портфеля"""
        current_price = self.exchange.get_current_price(self.symbol)
        
        if self.position == 1:
            position_value = self.position_size * current_price
            total_value = self.balance + position_value
            unrealized_pnl = (current_price - self.entry_price) * self.position_size
        else:
            total_value = self.balance
            unrealized_pnl = 0
        
        total_return = ((total_value - self.initial_balance) / self.initial_balance) * 100
        
        return {
            'balance': self.balance,
            'position_size': self.position_size,
            'entry_price': self.entry_price,
            'current_price': current_price,
            'total_value': total_value,
            'unrealized_pnl': unrealized_pnl,
            'total_return': total_return
        }

def main():
    """Демонстрация торгового бота"""
    print("🤖 ДЕМОНСТРАЦИЯ КРИПТОВАЛЮТНОГО ТОРГОВОГО БОТА")
    print("=" * 60)
    
    # Инициализация биржи (без API ключей для демонстрации)
    exchange = BinanceAPI()
    
    # Получаем исторические данные для тестирования
    print("📊 Получение данных с Binance...")
    data = exchange.get_market_data("BTC/USDT", interval="1h", limit=100)
    
    if not data.empty:
        print(f"✅ Получено {len(data)} записей данных")
        print(f"📈 Последняя цена: ${data['close'].iloc[-1]:.2f}")
        
        # Создаем стратегию
        strategy = MovingAverageStrategy(fast_period=10, slow_period=20, initial_capital=10000)
        
        # Создаем бота для бумажной торговли
        bot = PaperTradingBot(strategy, exchange, "BTC/USDT", initial_balance=10000)
        
        print("\n🎯 Настройка завершена!")
        print("📝 Для запуска реального бота:")
        print("   1. Получите API ключи с биржи")
        print("   2. Установите их в BinanceAPI")
        print("   3. Используйте LiveTradingBot вместо PaperTradingBot")
        print("   4. Запустите bot.start_trading()")
        
        # Демонстрация проверки сигналов
        print("\n🔍 Проверка текущих сигналов...")
        bot.check_signals()
        
        # Показываем статус портфеля
        status = bot.get_portfolio_status()
        print(f"\n💼 СТАТУС ПОРТФЕЛЯ:")
        print(f"   💰 Баланс: ${status['balance']:.2f}")
        print(f"   📊 Общая стоимость: ${status['total_value']:.2f}")
        print(f"   📈 Доходность: {status['total_return']:.2f}%")
        
    else:
        print("❌ Не удалось получить данные с биржи")
        print("🌐 Проверьте подключение к интернету")

if __name__ == "__main__":
    main()