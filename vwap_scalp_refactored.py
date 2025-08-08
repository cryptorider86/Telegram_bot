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
try:
    import ccxt  # type: ignore
except Exception:  # pragma: no cover
    ccxt = None


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
class Trade:
    """Internal trade representation for managing SL/TP and PnL."""
    symbol: str
    side: OrderType
    quantity: float
    entry_price: float
    stop_loss: float
    take_profit: float
    opened_at: datetime
    closed_at: Optional[datetime] = None
    is_open: bool = True
    realized_pnl: float = 0.0


@dataclass
class VWAPConfig:
    """Configuration for VWAP calculation."""
    lookback_period: int = 20
    volume_threshold: float = 1000.0
    price_deviation_threshold: float = 0.02
    min_trade_interval: int = 60  # seconds
    std_dev: float = 2.0
    risk_per_trade_usd: float = 50.0
    max_open_trades_per_symbol: int = 1
    take_profit_multiple: float = 1.0
    stop_loss_multiple: float = 1.0
    fee_rate: float = 0.0005  # 5 bps per side
    slippage_bps: float = 2.0  # 2 bps slippage per side
    exit_on_vwap_touch: bool = True


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
            df = df.copy()
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
            df = df.copy()
            df['typical_price'] = (df['high'] + df['low'] + df['close']) / 3
            price_std = df['typical_price'].std()
            
            upper_band = vwap + (std_dev * price_std)
            lower_band = vwap - (std_dev * price_std)
            
            return vwap, upper_band, lower_band
        
        except Exception as e:
            self.logger.error(f"Error calculating VWAP bands: {e}")
            return 0.0, 0.0, 0.0
    
    def calculate_volatility(self, df: pd.DataFrame) -> float:
        """Estimate volatility using std of typical price over lookback."""
        try:
            if df.empty:
                return 0.0
            df = df.copy()
            df['typical_price'] = (df['high'] + df['low'] + df['close']) / 3
            return float(df['typical_price'].std())
        except Exception as e:
            self.logger.error(f"Error calculating volatility: {e}")
            return 0.0


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
            
            # Enforce volume threshold on latest candle
            latest_volume = float(df['volume'].iloc[-1]) if 'volume' in df.columns else 0.0
            if latest_volume < self.config.volume_threshold:
                return None
            
            # Calculate VWAP and bands
            vwap, upper_band, lower_band = self.vwap_calculator.calculate_vwap_bands(
                df, std_dev=self.config.std_dev
            )
            
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
    
    def __init__(self, max_position_size: float, max_daily_loss: float, max_drawdown: float, initial_balance: float = 10000.0):
        self.max_position_size = max_position_size
        self.max_daily_loss = max_daily_loss
        self.max_drawdown = max_drawdown
        self.initial_balance = initial_balance
        self.daily_pnl = 0.0
        self.peak_equity = initial_balance
        self.logger = logging.getLogger(__name__)
    
    def current_equity(self) -> float:
        return self.initial_balance + self.daily_pnl
    
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
            # Check position size (exposure in quote currency)
            total_exposure = sum(abs(pos.quantity) * pos.avg_price for pos in current_positions)
            if total_exposure + (order.quantity * order.price) > self.max_position_size:
                self.logger.warning("Trade rejected: Position size limit exceeded")
                return False
            
            # Check daily loss limit
            if self.daily_pnl < -self.max_daily_loss:
                self.logger.warning("Trade rejected: Daily loss limit exceeded")
                return False
            
            # Check drawdown based on equity
            equity = self.current_equity()
            if self.peak_equity <= 0:
                self.peak_equity = equity
            drawdown = (self.peak_equity - equity) / self.peak_equity if self.peak_equity > 0 else 0.0
            if drawdown > self.max_drawdown:
                self.logger.warning("Trade rejected: Maximum drawdown exceeded")
                return False
            
            return True
        
        except Exception as e:
            self.logger.error(f"Error in risk check: {e}")
            return False
    
    def update_pnl(self, pnl: float):
        """Update daily PnL and peak equity."""
        self.daily_pnl += pnl
        equity = self.current_equity()
        if equity > self.peak_equity:
            self.peak_equity = equity
    
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
        self.open_trades: Dict[str, List[Trade]] = {}
        self._last_risk_reset_date: Optional[datetime] = None
    
    def should_trade(self, symbol: str) -> bool:
        """Check if enough time has passed since last trade."""
        if symbol not in self.last_trade_time:
            return True
        
        time_since_last_trade = time.time() - self.last_trade_time[symbol]
        return time_since_last_trade >= self.config.min_trade_interval
    
    def _maybe_reset_daily_risk(self):
        """Reset daily risk metrics when calendar day changes."""
        now = datetime.utcnow().date()
        if self._last_risk_reset_date is None:
            self._last_risk_reset_date = now
            return
        if now != self._last_risk_reset_date:
            self.risk_manager.reset_daily_stats()
            self._last_risk_reset_date = now
            self.logger.info("Daily risk stats reset")
    
    def _apply_slippage(self, side: OrderType, price: float) -> float:
        """Apply slippage to a price based on side."""
        bps = self.config.slippage_bps / 10000.0
        if side == OrderType.BUY:
            return price * (1 + bps)
        return price * (1 - bps)
    
    def _estimate_fees(self, entry_price: float, exit_price: float, quantity: float) -> float:
        notional_entry = abs(entry_price * quantity)
        notional_exit = abs(exit_price * quantity)
        return self.config.fee_rate * (notional_entry + notional_exit)
    
    def execute_signal(self, symbol: str, signal: OrderType, current_price: float, df: pd.DataFrame) -> bool:
        """
        Execute a trading signal.
        
        Args:
            symbol: Trading symbol
            signal: Trading signal
            current_price: Current market price
            df: Latest OHLCV window
            
        Returns:
            True if order placed successfully, False otherwise
        """
        try:
            # Respect per-symbol open trades constraint
            open_trades_for_symbol = self.open_trades.get(symbol, [])
            if len([t for t in open_trades_for_symbol if t.is_open]) >= self.config.max_open_trades_per_symbol:
                self.logger.info(f"Trade skipped for {symbol}: Max open trades reached")
                return False
            
            # Check if we should trade
            if not self.should_trade(symbol):
                self.logger.info(f"Trade skipped for {symbol}: Minimum interval not met")
                return False
            
            # Volatility-based sizing
            volatility = self.signal_generator.vwap_calculator.calculate_volatility(df)
            stop_distance = max(volatility * self.config.stop_loss_multiple, 1e-8)
            if stop_distance == 0:
                self.logger.info(f"Trade skipped for {symbol}: Zero volatility")
                return False
            
            # Risk-based quantity (in quote currency)
            quantity = (self.config.risk_per_trade_usd / stop_distance)
            # Ensure notional does not exceed risk manager's max position size per single order
            max_qty_by_exposure = self.risk_manager.max_position_size / max(current_price, 1e-8)
            quantity = float(max(0.0, min(quantity, max_qty_by_exposure)))
            if quantity <= 0:
                self.logger.info(f"Trade skipped for {symbol}: Computed quantity <= 0")
                return False
            
            # Define SL/TP around entry
            entry_price = current_price
            if signal == OrderType.BUY:
                stop_loss = entry_price - stop_distance
                take_profit = entry_price + self.config.take_profit_multiple * stop_distance
            else:
                stop_loss = entry_price + stop_distance
                take_profit = entry_price - self.config.take_profit_multiple * stop_distance
            
            # Create order
            order = Order(
                symbol=symbol,
                order_type=signal,
                quantity=quantity,
                price=entry_price,
                timestamp=datetime.now()
            )
            
            # Risk check (use exchange positions for exposure if available)
            current_positions = list(self.positions.values())
            if not self.risk_manager.can_trade(order, current_positions):
                return False
            
            # Place order
            order_id = self.exchange.place_order(order)
            if order_id:
                order.order_id = order_id
                self.active_orders[order_id] = order
                self.last_trade_time[symbol] = time.time()
                
                # Track trade locally
                trade = Trade(
                    symbol=symbol,
                    side=signal,
                    quantity=quantity,
                    entry_price=self._apply_slippage(signal, entry_price),
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    opened_at=datetime.now()
                )
                self.open_trades.setdefault(symbol, []).append(trade)
                
                self.logger.info(
                    f"Order placed: {signal.value} qty={quantity:.6f} {symbol} @ {entry_price:.4f} SL={stop_loss:.4f} TP={take_profit:.4f}"
                )
                return True
            
            return False
        
        except Exception as e:
            self.logger.error(f"Error executing signal: {e}")
            return False
    
    def close_trade(self, symbol: str, trade: Trade, exit_price: float):
        """Close a trade, place an opposite order, and update PnL."""
        try:
            if not trade.is_open:
                return
            
            # Place opposite side order to close
            exit_side = OrderType.SELL if trade.side == OrderType.BUY else OrderType.BUY
            exit_order = Order(
                symbol=symbol,
                order_type=exit_side,
                quantity=trade.quantity,
                price=exit_price,
                timestamp=datetime.now()
            )
            order_id = self.exchange.place_order(exit_order)
            if not order_id:
                self.logger.warning(f"Failed to close trade for {symbol}")
                return
            
            filled_exit_price = self._apply_slippage(exit_side, exit_price)
            gross_pnl = (filled_exit_price - trade.entry_price) * trade.quantity
            if trade.side == OrderType.SELL:
                gross_pnl = -gross_pnl
            fees = self._estimate_fees(trade.entry_price, filled_exit_price, trade.quantity)
            realized_pnl = gross_pnl - fees
            
            trade.is_open = False
            trade.closed_at = datetime.now()
            trade.realized_pnl = realized_pnl
            
            self.risk_manager.update_pnl(realized_pnl)
            self.logger.info(
                f"Closed {symbol} {trade.side.value} qty={trade.quantity:.6f} @ {filled_exit_price:.4f} PnL={realized_pnl:.4f}"
            )
        except Exception as e:
            self.logger.error(f"Error closing trade: {e}")
    
    def manage_exits(self, symbol: str, current_price: float, df: pd.DataFrame):
        """Check SL/TP and optional VWAP-touch exits for open trades."""
        try:
            if symbol not in self.open_trades:
                return
            vwap = 0.0
            if self.config.exit_on_vwap_touch:
                vwap = self.signal_generator.vwap_calculator.calculate_vwap(df)
            
            for trade in list(self.open_trades[symbol]):
                if not trade.is_open:
                    continue
                
                exit_reason = None
                if trade.side == OrderType.BUY:
                    if current_price <= trade.stop_loss:
                        exit_reason = "SL"
                    elif current_price >= trade.take_profit:
                        exit_reason = "TP"
                    elif self.config.exit_on_vwap_touch and vwap > 0 and current_price >= vwap:
                        exit_reason = "VWAP"
                else:
                    if current_price >= trade.stop_loss:
                        exit_reason = "SL"
                    elif current_price <= trade.take_profit:
                        exit_reason = "TP"
                    elif self.config.exit_on_vwap_touch and vwap > 0 and current_price <= vwap:
                        exit_reason = "VWAP"
                
                if exit_reason:
                    self.logger.info(f"Exit {exit_reason} triggered for {symbol}")
                    self.close_trade(symbol, trade, exit_price=current_price)
        except Exception as e:
            self.logger.error(f"Error managing exits: {e}")
    
    def update_positions(self):
        """Update current positions from exchange."""
        try:
            positions = self.exchange.get_positions()
            self.positions = {pos.symbol: pos for pos in positions}
        except Exception as e:
            self.logger.error(f"Error updating positions: {e}")
    
    def run_strategy(self, symbols: List[str], timeframe: str = "1m", max_iterations: Optional[int] = None, sleep_seconds: float = 10.0):
        """
        Run the VWAP scalping strategy.
        
        Args:
            symbols: List of symbols to trade
            timeframe: Data timeframe
            max_iterations: Optional limit for main loop iterations (testing)
            sleep_seconds: Sleep between iterations
        """
        self.logger.info("Starting VWAP Scalping Strategy")
        
        try:
            iterations = 0
            while True:
                # Reset daily risk if needed
                self._maybe_reset_daily_risk()
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
                            self.execute_signal(symbol, signal, current_price, df)
                        
                        # Manage exits for open trades
                        self.manage_exits(symbol, current_price, df)
                        
                        # Update positions
                        self.update_positions()
                        
                    except Exception as e:
                        self.logger.error(f"Error processing {symbol}: {e}")
                        continue
                
                # Sleep between iterations
                iterations += 1
                if max_iterations is not None and iterations >= max_iterations:
                    self.logger.info("Max iterations reached; stopping strategy loop")
                    break
                time.sleep(sleep_seconds)
        
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


class CCXTDataProvider(DataProvider):
    """Data provider using CCXT exchange clients (e.g., OKX, MEXC)."""
    def __init__(self, exchange_id: str, market_type: str = "spot", enable_rate_limit: bool = True):
        if ccxt is None:
            raise RuntimeError("ccxt is not installed. Please install ccxt to use CCXTDataProvider.")
        self.exchange_id = exchange_id
        self.market_type = market_type
        self.logger = logging.getLogger(__name__)
        cls = getattr(ccxt, exchange_id)
        self.exchange = cls({
            'enableRateLimit': enable_rate_limit,
            'options': {
                'defaultType': market_type
            }
        })
        try:
            self.exchange.load_markets()
        except Exception as e:
            self.logger.warning(f"Failed to load markets for {exchange_id}: {e}")
        
    def _build_df(self, ohlcv: List[List[Any]]) -> pd.DataFrame:
        if not ohlcv:
            return pd.DataFrame(columns=['open', 'high', 'low', 'close', 'volume'])
        arr = np.array(ohlcv)
        # CCXT OHLCV: [timestamp, open, high, low, close, volume]
        ts = pd.to_datetime(arr[:, 0], unit='ms')
        df = pd.DataFrame({
            'open': arr[:, 1].astype(float),
            'high': arr[:, 2].astype(float),
            'low': arr[:, 3].astype(float),
            'close': arr[:, 4].astype(float),
            'volume': arr[:, 5].astype(float),
        }, index=ts)
        return df
    
    def get_ohlcv(self, symbol: str, timeframe: str, limit: int) -> pd.DataFrame:
        try:
            # Many exchanges accept '1m' as timeframe
            candles = self.exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
            return self._build_df(candles)
        except Exception as e:
            self.logger.warning(f"{self.exchange_id} get_ohlcv failed for {symbol}: {e}")
            return pd.DataFrame(columns=['open', 'high', 'low', 'close', 'volume'])
    
    def get_current_price(self, symbol: str) -> float:
        try:
            ticker = self.exchange.fetch_ticker(symbol)
            last = ticker.get('last') or ticker.get('close') or 0.0
            return float(last) if last else 0.0
        except Exception as e:
            self.logger.warning(f"{self.exchange_id} get_current_price failed for {symbol}: {e}")
            return 0.0


class CompositeDataProvider(DataProvider):
    """Tries multiple providers in order, returns first successful result."""
    def __init__(self, providers: List[DataProvider]):
        self.providers = providers
        self.logger = logging.getLogger(__name__)
    
    def get_ohlcv(self, symbol: str, timeframe: str, limit: int) -> pd.DataFrame:
        for p in self.providers:
            df = p.get_ohlcv(symbol, timeframe, limit)
            if not df.empty:
                return df
        self.logger.warning(f"All providers failed to fetch OHLCV for {symbol}")
        return pd.DataFrame(columns=['open', 'high', 'low', 'close', 'volume'])
    
    def get_current_price(self, symbol: str) -> float:
        for p in self.providers:
            price = p.get_current_price(symbol)
            if price and price > 0:
                return price
        self.logger.warning(f"All providers failed to fetch price for {symbol}")
        return 0.0


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
        
        # Update positions (very naive: append legs)
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
            min_trade_interval=60,
            std_dev=2.0,
            risk_per_trade_usd=50.0,
            max_open_trades_per_symbol=1,
            take_profit_multiple=1.0,
            stop_loss_multiple=1.0,
            fee_rate=0.0005,
            slippage_bps=2.0,
            exit_on_vwap_touch=True
        )
        
        # Risk management
        risk_manager = RiskManager(
            max_position_size=10000.0,
            max_daily_loss=1000.0,
            max_drawdown=0.1,
            initial_balance=10000.0
        )
        
        # Initialize components
        data_providers: List[DataProvider] = []
        try:
            if ccxt is not None:
                # OKX and MEXC spot as data sources
                data_providers.append(CCXTDataProvider('okx', market_type='spot'))
                data_providers.append(CCXTDataProvider('mexc', market_type='spot'))
        except Exception as e:
            logger.warning(f"Failed to init CCXT providers: {e}")
        # Always add mock as fallback
        data_providers.append(MockDataProvider())
        data_provider = CompositeDataProvider(data_providers)
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
        import os
        max_iters_env = os.getenv("VWAP_MAX_ITERS")
        max_iters = int(max_iters_env) if max_iters_env else None
        strategy.run_strategy(symbols, max_iterations=max_iters, sleep_seconds=2.0)
    
    except Exception as e:
        logger.error(f"Main error: {e}")


if __name__ == "__main__":
    main()