# 🚀 Система Торговых Стратегий

Комплексная система для разработки, тестирования и автоматизации торговых стратегий с поддержкой различных финансовых рынков.

## 📋 Возможности

### 🎯 Торговые Стратегии
- **Moving Average Strategy** - Стратегия на основе скользящих средних
- **RSI Strategy** - Стратегия на основе индикатора RSI
- **Bollinger Bands Strategy** - Стратегия на основе полос Боллинджера
- **Расширяемая архитектура** для создания собственных стратегий

### 📊 Технические Индикаторы
- Простая скользящая средняя (SMA)
- Экспоненциальная скользящая средняя (EMA)
- Индекс относительной силы (RSI)
- Полосы Боллинджера
- MACD индикатор

### 🤖 Автоматизация
- **Живая торговля** с API биржи
- **Бумажная торговля** для тестирования без риска
- **Backtesting** на исторических данных
- Уведомления о торговых сигналах

### 📈 Анализ Производительности
- Расчет доходности
- Максимальная просадка
- Коэффициент Шарпа
- Процент прибыльных сделок
- Визуализация результатов

## 🛠️ Установка

1. **Клонируйте репозиторий**
```bash
git clone <repository-url>
cd <repository-name>
```

2. **Установите зависимости**
```bash
pip install -r requirements.txt
```

## 🚀 Быстрый старт

### 1. Базовое тестирование стратегий

```python
from trading_strategy import MovingAverageStrategy, Backtester, get_sample_data

# Получаем тестовые данные
data = get_sample_data()

# Создаем стратегию
strategy = MovingAverageStrategy(fast_period=10, slow_period=30)

# Запускаем бэктестинг
backtester = Backtester(strategy)
results = backtester.run_backtest(data)

print(f"Доходность: {results['total_return']:.2%}")
```

### 2. Демонстрация всех стратегий

```bash
python trading_strategy.py
```

### 3. Криптовалютный торговый бот

```bash
python crypto_trading_bot.py
```

## 📚 Подробная документация

### Создание собственной стратегии

```python
from trading_strategy import TradingStrategy
import pandas as pd

class MyCustomStrategy(TradingStrategy):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Инициализация параметров стратегии
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()
        
        # Ваша логика генерации сигналов
        df['signal'] = 0  # 0 = держать, 1 = покупать, -1 = продавать
        
        # Пример простой логики
        df.loc[df['close'] > df['close'].shift(1), 'signal'] = 1
        df.loc[df['close'] < df['close'].shift(1), 'signal'] = -1
        
        return df
```

### Интеграция с реальной биржей

Для работы с реальной биржей необходимо:

1. **Получить API ключи** с выбранной биржи
2. **Настроить API клиент**:

```python
from crypto_trading_bot import BinanceAPI, LiveTradingBot
from trading_strategy import MovingAverageStrategy

# Настройка API (ОСТОРОЖНО: не публикуйте ключи!)
exchange = BinanceAPI(
    api_key="your_api_key_here",
    api_secret="your_api_secret_here"
)

# Создание стратегии
strategy = MovingAverageStrategy(fast_period=10, slow_period=20)

# Создание и запуск бота
bot = LiveTradingBot(strategy, exchange, "BTC/USDT")
bot.start_trading(check_interval=300)  # Проверка каждые 5 минут
```

### Бумажная торговля

Для безопасного тестирования без реальных денег:

```python
from crypto_trading_bot import BinanceAPI, PaperTradingBot

exchange = BinanceAPI()  # Без API ключей
bot = PaperTradingBot(strategy, exchange, "BTC/USDT", initial_balance=10000)

# Однократная проверка сигналов
bot.check_signals()

# Статус портфеля
status = bot.get_portfolio_status()
print(f"Доходность: {status['total_return']:.2f}%")
```

## ⚙️ Конфигурация

### Параметры стратегий

**Moving Average Strategy:**
- `fast_period` - Период быстрой скользящей средней (по умолчанию: 10)
- `slow_period` - Период медленной скользящей средней (по умолчанию: 30)

**RSI Strategy:**
- `rsi_period` - Период расчета RSI (по умолчанию: 14)
- `oversold` - Уровень перепроданности (по умолчанию: 30)
- `overbought` - Уровень перекупленности (по умолчанию: 70)

**Bollinger Bands Strategy:**
- `bb_period` - Период расчета полос (по умолчанию: 20)
- `bb_std` - Количество стандартных отклонений (по умолчанию: 2)

### Управление рисками

```python
strategy = MovingAverageStrategy(
    initial_capital=10000,
    commission=0.001,  # 0.1% комиссия
)

# Расчет размера позиции (2% риска от капитала)
position_size = strategy.calculate_position_size(price, risk_percent=0.02)
```

## 📊 Поддерживаемые биржи

### Текущие
- **Binance** - Полная поддержка API

### Планируемые
- Bybit
- OKX
- Coinbase
- Kraken

## ⚠️ Важные предупреждения

1. **Торговля сопряжена с рисками** - можете потерять все инвестиции
2. **Тестируйте стратегии** на исторических данных и бумажной торговле
3. **Начинайте с малых сумм** при реальной торговле
4. **Не публикуйте API ключи** в открытом коде
5. **Используйте stop-loss** для ограничения убытков

## 🔧 Разработка

### Структура проекта

```
├── trading_strategy.py      # Основной модуль стратегий
├── crypto_trading_bot.py    # Торговый бот для криптовалют
├── requirements.txt         # Зависимости
├── README.md               # Документация
└── examples/               # Примеры использования
```

### Добавление новых индикаторов

```python
class TechnicalIndicators:
    @staticmethod
    def your_indicator(data: pd.Series, period: int) -> pd.Series:
        # Ваша реализация индикатора
        return result
```

## 📝 Лицензия

MIT License - см. файл LICENSE для деталей.

## 🤝 Поддержка

Если у вас есть вопросы или предложения:
1. Создайте Issue в репозитории
2. Отправьте Pull Request с улучшениями
3. Напишите в Telegram: @your_username

## 📈 Примеры результатов

После запуска `python trading_strategy.py`:

```
🚀 Демонстрация торговых стратегий
==================================================
📊 Загружено 366 дней данных

📈 Тестирование стратегии: Moving Average (10/30)
💰 Итоговая доходность: 15.23%
💵 Финальная стоимость: $11523.45
📉 Максимальная просадка: 8.45%
📊 Коэффициент Шарпа: 1.23
🎯 Процент прибыльных сделок: 65.00%
🔄 Всего сделок: 42

==================================================
📊 СРАВНЕНИЕ СТРАТЕГИЙ
==================================================
🏆 Лучшая стратегия: Moving Average (10/30)
🎯 Доходность: 15.23%
```
