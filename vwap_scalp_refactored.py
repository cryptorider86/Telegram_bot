#!/usr/bin/env python3
"""
VWAP Scalping Strategy - Refactored Version
A modular and clean implementation of a VWAP-based scalping trading strategy.
"""

import logging
import time
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import pandas as pd
import numpy as np
from abc import ABC, abstractmethod


class OrderType(Enum):
    """Order types for trading."""
    BUY = "BUY"
    SELL = "SELL"


class OrderStatus(Enum):
    """Order status enumeration."""
    PENDING = "PENDING"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"


@dataclass
class Order:
    """Order data structure."""
    symbol: str
    order_type: OrderType
    quantity: float
    price: float
    timestamp: datetime
    status: OrderStatus = OrderStatus.PENDING
    order_id: Optional[str] = None


@dataclass
class Position:
    """Position data structure."""
    symbol: str
    quantity: float
    avg_price: float
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0


@dataclass
class VWAPConfig:
    """Configuration for VWAP calculation."""
    lookback_period: int = 20
    volume_threshold: float = 1000.0
    price_deviation_threshold: float = 0.02
    min_trade_interval: int = 60  # seconds


class DataProvider(ABC):
    """Abstract base class for data providers."""
    
    @abstractmethod
    def get_ohlcv(self, symbol: str, timeframe: str, limit: int) -> pd.DataFrame:
        """Get OHLCV data for a symbol."""
        pass
    
    @abstractmethod
    def get_current_price(self, symbol: str) -> float:
        """Get current price for a symbol."""
        pass


class ExchangeInterface(ABC):
    """Abstract base class for exchange interfaces."""
    
    @abstractmethod
    def place_order(self, order: Order) -> str:
        """Place an order and return order ID."""
        pass
    
    @abstractmethod
    def cancel_order(self, order_id: str) -> bool:
        """Cancel an order."""
        pass
    
    @abstractmethod
    def get_order_status(self, order_id: str) -> OrderStatus:
        """Get order status."""
        pass
    
    @abstractmethod
    def get_positions(self) -> List[Position]:
        """Get current positions."""
        pass


class VWAPCalculator:
    """VWAP calculation utility."""
    
    def __init__(self, config: VWAPConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def calculate_vwap(self, df: pd.DataFrame) -> float:
        """
        Calculate VWAP (Volume Weighted Average Price).
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            VWAP value
        """
        try:
            if df.empty:
                return 0.0
            
            # Calculate typical price
            df['typical_price'] = (df['high'] + df['low'] + df['close']) / 3
            
            # Calculate volume-weighted price
            vwap = (df['typical_price'] * df['volume']).sum() / df['volume'].sum()
            
            return float(vwap)
        
        except Exception as e:
            self.logger.error(f"Error calculating VWAP: {e}")
            return 0.0
    
    def calculate_vwap_bands(self, df: pd.DataFrame, std_dev: float = 2.0) -> Tuple[float, float, float]:
        """
        Calculate VWAP with upper and lower bands.
        
        Args:
            df: DataFrame with OHLCV data
            std_dev: Standard deviation multiplier for bands
            
        Returns:
            Tuple of (vwap, upper_band, lower_band)
        """
        try:
            vwap = self.calculate_vwap(df)
            
            if vwap == 0.0:
                return 0.0, 0.0, 0.0
            
            # Calculate price volatility
            df['typical_price'] = (df['high'] + df['low'] + df['close']) / 3
            price_std = df['typical_price'].std()
            
            upper_band = vwap + (std_dev * price_std)
            lower_band = vwap - (std_dev * price_std)
            
            return vwap, upper_band, lower_band
        
        except Exception as e:
            self.logger.error(f"Error calculating VWAP bands: {e}")
            return 0.0, 0.0, 0.0


class SignalGenerator:
    """Generate trading signals based on VWAP analysis."""
    
    def __init__(self, config: VWAPConfig):
        self.config = config
        self.vwap_calculator = VWAPCalculator(config)
        self.logger = logging.getLogger(__name__)
    
    def generate_signal(self, df: pd.DataFrame, current_price: float) -> Optional[OrderType]:
        """
        Generate trading signal based on VWAP analysis.
        
        Args:
            df: Historical OHLCV data
            current_price: Current market price
            
        Returns:
            OrderType if signal generated, None otherwise
        """
        try:
            if df.empty or len(df) < self.config.lookback_period:
                return None
            
            # Calculate VWAP and bands
            vwap, upper_band, lower_band = self.vwap_calculator.calculate_vwap_bands(df)
            
            if vwap == 0.0:
                return None
            
            # Calculate price deviation from VWAP
            price_deviation = abs(current_price - vwap) / vwap
            
            # Check if price is near VWAP bands
            near_upper = current_price >= upper_band * (1 - self.config.price_deviation_threshold)
            near_lower = current_price <= lower_band * (1 + self.config.price_deviation_threshold)
            
            # Generate signals
            if near_upper and price_deviation > self.config.price_deviation_threshold:
                return OrderType.SELL
            elif near_lower and price_deviation > self.config.price_deviation_threshold:
                return OrderType.BUY
            
            return None
        
        except Exception as e:
            self.logger.error(f"Error generating signal: {e}")
            return None


class RiskManager:
    """Risk management for the trading strategy."""
    
    def __init__(self, max_position_size: float, max_daily_loss: float, max_drawdown: float):
        self.max_position_size = max_position_size
        self.max_daily_loss = max_daily_loss
        self.max_drawdown = max_drawdown
        self.daily_pnl = 0.0
        self.peak_balance = 0.0
        self.logger = logging.getLogger(__name__)
    
    def can_trade(self, order: Order, current_positions: List[Position]) -> bool:
        """
        Check if a trade is allowed based on risk parameters.
        
        Args:
            order: Proposed order
            current_positions: Current open positions
            
        Returns:
            True if trade is allowed, False otherwise
        """
        try:
            # Check position size
            total_exposure = sum(abs(pos.quantity) * pos.avg_price for pos in current_positions)
            if total_exposure + (order.quantity * order.price) > self.max_position_size:
                self.logger.warning("Trade rejected: Position size limit exceeded")
                return False
            
            # Check daily loss limit
            if self.daily_pnl < -self.max_daily_loss:
                self.logger.warning("Trade rejected: Daily loss limit exceeded")
                return False
            
            # Check drawdown
            current_drawdown = (self.peak_balance - (self.peak_balance + self.daily_pnl)) / self.peak_balance
            if current_drawdown > self.max_drawdown:
                self.logger.warning("Trade rejected: Maximum drawdown exceeded")
                return False
            
            return True
        
        except Exception as e:
            self.logger.error(f"Error in risk check: {e}")
            return False
    
    def update_pnl(self, pnl: float):
        """Update daily PnL."""
        self.daily_pnl += pnl
        if self.peak_balance + self.daily_pnl > self.peak_balance:
            self.peak_balance = self.peak_balance + self.daily_pnl
    
    def reset_daily_stats(self):
        """Reset daily statistics."""
        self.daily_pnl = 0.0


class VWAPScalpStrategy:
    """Main VWAP scalping strategy class."""
    
    def __init__(self, 
                 data_provider: DataProvider,
                 exchange: ExchangeInterface,
                 config: VWAPConfig,
                 risk_manager: RiskManager):
        self.data_provider = data_provider
        self.exchange = exchange
        self.config = config
        self.risk_manager = risk_manager
        self.signal_generator = SignalGenerator(config)
        self.logger = logging.getLogger(__name__)
        
        # Strategy state
        self.last_trade_time = {}
        self.active_orders = {}
        self.positions = {}
    
    def should_trade(self, symbol: str) -> bool:
        """Check if enough time has passed since last trade."""
        if symbol not in self.last_trade_time:
            return True
        
        time_since_last_trade = time.time() - self.last_trade_time[symbol]
        return time_since_last_trade >= self.config.min_trade_interval
    
    def execute_signal(self, symbol: str, signal: OrderType, current_price: float) -> bool:
        """
        Execute a trading signal.
        
        Args:
            symbol: Trading symbol
            signal: Trading signal
            current_price: Current market price
            
        Returns:
            True if order placed successfully, False otherwise
        """
        try:
            # Check if we should trade
            if not self.should_trade(symbol):
                self.logger.info(f"Trade skipped for {symbol}: Minimum interval not met")
                return False
            
            # Calculate position size (simplified - could be more sophisticated)
            position_size = 1.0  # Fixed size for simplicity
            
            # Create order
            order = Order(
                symbol=symbol,
                order_type=signal,
                quantity=position_size,
                price=current_price,
                timestamp=datetime.now()
            )
            
            # Risk check
            current_positions = list(self.positions.values())
            if not self.risk_manager.can_trade(order, current_positions):
                return False
            
            # Place order
            order_id = self.exchange.place_order(order)
            if order_id:
                order.order_id = order_id
                self.active_orders[order_id] = order
                self.last_trade_time[symbol] = time.time()
                
                self.logger.info(f"Order placed: {signal.value} {position_size} {symbol} @ {current_price}")
                return True
            
            return False
        
        except Exception as e:
            self.logger.error(f"Error executing signal: {e}")
            return False
    
    def update_positions(self):
        """Update current positions from exchange."""
        try:
            positions = self.exchange.get_positions()
            self.positions = {pos.symbol: pos for pos in positions}
        except Exception as e:
            self.logger.error(f"Error updating positions: {e}")
    
    def run_strategy(self, symbols: List[str], timeframe: str = "1m"):
        """
        Run the VWAP scalping strategy.
        
        Args:
            symbols: List of symbols to trade
            timeframe: Data timeframe
        """
        self.logger.info("Starting VWAP Scalping Strategy")
        
        try:
            while True:
                for symbol in symbols:
                    try:
                        # Get market data
                        df = self.data_provider.get_ohlcv(symbol, timeframe, self.config.lookback_period)
                        current_price = self.data_provider.get_current_price(symbol)
                        
                        if df.empty or current_price == 0:
                            continue
                        
                        # Generate signal
                        signal = self.signal_generator.generate_signal(df, current_price)
                        
                        if signal:
                            self.logger.info(f"Signal generated for {symbol}: {signal.value}")
                            self.execute_signal(symbol, signal, current_price)
                        
                        # Update positions
                        self.update_positions()
                        
                    except Exception as e:
                        self.logger.error(f"Error processing {symbol}: {e}")
                        continue
                
                # Sleep between iterations
                time.sleep(10)
        
        except KeyboardInterrupt:
            self.logger.info("Strategy stopped by user")
        except Exception as e:
            self.logger.error(f"Strategy error: {e}")


class MockDataProvider(DataProvider):
    """Mock data provider for testing."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def get_ohlcv(self, symbol: str, timeframe: str, limit: int) -> pd.DataFrame:
        """Mock OHLCV data."""
        # Generate mock data
        dates = pd.date_range(end=datetime.now(), periods=limit, freq='1min')
        data = {
            'open': np.random.uniform(100, 200, limit),
            'high': np.random.uniform(100, 200, limit),
            'low': np.random.uniform(100, 200, limit),
            'close': np.random.uniform(100, 200, limit),
            'volume': np.random.uniform(1000, 5000, limit)
        }
        return pd.DataFrame(data, index=dates)
    
    def get_current_price(self, symbol: str) -> float:
        """Mock current price."""
        return np.random.uniform(100, 200)


class MockExchange(ExchangeInterface):
    """Mock exchange interface for testing."""
    
    def __init__(self):
        self.orders = {}
        self.positions = []
        self.order_counter = 0
        self.logger = logging.getLogger(__name__)
    
    def place_order(self, order: Order) -> str:
        """Mock order placement."""
        order_id = f"order_{self.order_counter}"
        self.order_counter += 1
        order.order_id = order_id
        order.status = OrderStatus.FILLED
        self.orders[order_id] = order
        
        # Update positions
        position = Position(
            symbol=order.symbol,
            quantity=order.quantity if order.order_type == OrderType.BUY else -order.quantity,
            avg_price=order.price
        )
        self.positions.append(position)
        
        self.logger.info(f"Mock order placed: {order_id}")
        return order_id
    
    def cancel_order(self, order_id: str) -> bool:
        """Mock order cancellation."""
        if order_id in self.orders:
            self.orders[order_id].status = OrderStatus.CANCELLED
            return True
        return False
    
    def get_order_status(self, order_id: str) -> OrderStatus:
        """Mock order status."""
        if order_id in self.orders:
            return self.orders[order_id].status
        return OrderStatus.REJECTED
    
    def get_positions(self) -> List[Position]:
        """Mock positions."""
        return self.positions


def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('vwap_scalp.log'),
            logging.StreamHandler()
        ]
    )


def main():
    """Main function to run the VWAP scalping strategy."""
    setup_logging()
    logger = logging.getLogger(__name__)
    
    try:
        # Configuration
        config = VWAPConfig(
            lookback_period=20,
            volume_threshold=1000.0,
            price_deviation_threshold=0.02,
            min_trade_interval=60
        )
        
        # Risk management
        risk_manager = RiskManager(
            max_position_size=10000.0,
            max_daily_loss=1000.0,
            max_drawdown=0.1
        )
        
        # Initialize components
        data_provider = MockDataProvider()
        exchange = MockExchange()
        
        # Create strategy
        strategy = VWAPScalpStrategy(
            data_provider=data_provider,
            exchange=exchange,
            config=config,
            risk_manager=risk_manager
        )
        
        # Run strategy
        symbols = ["BTC/USDT", "ETH/USDT"]
        strategy.run_strategy(symbols)
    
    except Exception as e:
        logger.error(f"Main error: {e}")


if __name__ == "__main__":
    main()