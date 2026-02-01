"""
===============================
TRADING BOT (FIXED VERSION)
===============================

PERBAIKAN DARI VERSI LAMA:
1. ✅ Feature consistency - Import dari features.py
2. ✅ Proper PnL calculation - Gunakan pos.profit dari MT5
3. ✅ Multiple position handling - Loop semua posisi
4. ✅ Better error handling
5. ✅ Optimal threshold usage
6. ✅ Connection health monitoring

CRITICAL: Bot ini HARUS ditest di DEMO account dulu!
"""

import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import joblib
from datetime import datetime, time, timedelta
import pytz
import warnings
import time as dt
import os
import logging
import json

# Import shared features
from features import add_all_features, get_feature_columns, validate_features

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('bot.log'),
        logging.StreamHandler()
    ]
)

warnings.filterwarnings('ignore')

# =====================================
# KONFIGURASI
# =====================================

SYMBOL = "XAUUSD"
TIMEFRAME = mt5.TIMEFRAME_H1
MAGIC_NUMBER = 12345
TIMEZONE = pytz.timezone("Asia/Jakarta")

# Risk Management
RISK_PERCENT = 2.0
MIN_LOT = 0.01
ATR_PERIOD = 14
RISK_MULTIPLIER = 1.5
REWARD_MULTIPLIER = 3.0
LOSS_COOLDOWN_MINUTES = 30

# Filters & Limits
TRADING_SESSION_START = time(13, 0)
TRADING_SESSION_END = time(23, 59)
MIDNIGHT_CLOSE_TIME = time(0, 0)

# CRITICAL: Load optimal thresholds from training
# Default jika metadata tidak ada
DEFAULT_BUY_THRESHOLD = 0.70
DEFAULT_SELL_THRESHOLD = 0.70

MAX_TRADES_PER_DAY = 3
MAX_DAILY_LOSS_PERCENT = 5.0
MAX_CONSECUTIVE_LOSSES = 3
MAX_DRAWDOWN_PERCENT = 15.0
MAX_SPREAD_MULTIPLIER = 3.0

# Files
MODEL_BUY_FILE = "model_buy.pkl"
MODEL_SELL_FILE = "model_sell.pkl"
MODEL_METADATA_FILE = "model_metadata.pkl"
JOURNAL_FILE = "trade_journal.csv"
STATE_FILE = "bot_state.json"

# Connection health
MAX_CONNECTION_ERRORS = 5
CONNECTION_CHECK_INTERVAL = 300  # 5 minutes


# =====================================
# STATE MANAGEMENT (FIXED)
# =====================================

class DailyState:
    """Enhanced state management dengan persistence"""
    
    def __init__(self):
        self.reset()
        self.connection_errors = 0
        self.last_connection_check = datetime.now(TIMEZONE)
    
    def reset(self):
        """Reset daily counters"""
        self.date = datetime.now(TIMEZONE).date()
        self.trades_today = 0
        self.daily_pnl = 0.0
        self.consecutive_losses = 0
        self.last_loss_time = None
        
        acc = mt5.account_info()
        self.initial_balance = acc.balance if acc else 0.0
        self.peak_balance = self._load_peak_balance()
        
        logging.info(f"State reset for {self.date}")
        logging.info(f"Initial balance: ${self.initial_balance:.2f}")
        logging.info(f"Peak balance: ${self.peak_balance:.2f}")
    
    def _load_peak_balance(self):
        """Load peak balance from state file"""
        try:
            if os.path.exists(STATE_FILE):
                with open(STATE_FILE, 'r') as f:
                    data = json.load(f)
                    return max(data.get('peak_balance', self.initial_balance), self.initial_balance)
        except Exception as e:
            logging.warning(f"Failed to load state: {e}")
        return self.initial_balance
    
    def _save_peak_balance(self):
        """Save peak balance to state file"""
        try:
            with open(STATE_FILE, 'w') as f:
                json.dump({'peak_balance': self.peak_balance}, f)
        except Exception as e:
            logging.error(f"Failed to save state: {e}")
    
    def update(self, pnl):
        """
        Update state after trade closes.
        
        CRITICAL FIX: Ini dipanggil dari monitor_positions dengan PnL yang benar
        """
        self.daily_pnl += pnl
        self.trades_today += 1
        
        if pnl < 0:
            self.consecutive_losses += 1
            self.last_loss_time = datetime.now(TIMEZONE)
            logging.warning(f"Loss recorded: ${pnl:.2f} (consecutive: {self.consecutive_losses})")
        else:
            self.consecutive_losses = 0
            logging.info(f"Win recorded: ${pnl:.2f}")
        
        # Update peak balance
        current_balance = self._get_current_balance()
        if current_balance > self.peak_balance:
            self.peak_balance = current_balance
            self._save_peak_balance()
            logging.info(f"New peak balance: ${self.peak_balance:.2f}")
        
        logging.info(f"Daily stats - Trades: {self.trades_today}, PnL: ${self.daily_pnl:.2f}")
    
    def _get_current_balance(self):
        """Get current account balance"""
        acc = mt5.account_info()
        return acc.balance if acc else 0.0
    
    def cooldown_active(self):
        """Check if in cooldown period after loss"""
        if not self.last_loss_time:
            return False
        
        cooldown_end = self.last_loss_time + timedelta(minutes=LOSS_COOLDOWN_MINUTES)
        is_active = datetime.now(TIMEZONE) < cooldown_end
        
        if is_active:
            remaining = (cooldown_end - datetime.now(TIMEZONE)).total_seconds() / 60
            logging.info(f"Cooldown active: {remaining:.1f} minutes remaining")
        
        return is_active
    
    def check_daily_limits(self):
        """
        Check if daily limits exceeded.
        
        Returns:
            tuple: (can_trade, reason)
        """
        # Check max trades
        if self.trades_today >= MAX_TRADES_PER_DAY:
            return False, f"Max trades reached ({MAX_TRADES_PER_DAY})"
        
        # Check max daily loss
        if self.daily_pnl < 0:
            loss_pct = abs(self.daily_pnl / self.initial_balance * 100)
            if loss_pct >= MAX_DAILY_LOSS_PERCENT:
                return False, f"Max daily loss reached ({loss_pct:.1f}%)"
        
        # Check consecutive losses
        if self.consecutive_losses >= MAX_CONSECUTIVE_LOSSES:
            return False, f"Max consecutive losses ({MAX_CONSECUTIVE_LOSSES})"
        
        # Check max drawdown
        current_balance = self._get_current_balance()
        drawdown_pct = ((self.peak_balance - current_balance) / self.peak_balance * 100)
        if drawdown_pct >= MAX_DRAWDOWN_PERCENT:
            return False, f"Max drawdown reached ({drawdown_pct:.1f}%)"
        
        return True, "OK"
    
    def check_connection_health(self):
        """Monitor MT5 connection health"""
        now = datetime.now(TIMEZONE)
        
        if (now - self.last_connection_check).total_seconds() < CONNECTION_CHECK_INTERVAL:
            return True
        
        self.last_connection_check = now
        
        # Try to get account info as health check
        acc = mt5.account_info()
        if acc is None:
            self.connection_errors += 1
            logging.error(f"Connection check failed ({self.connection_errors}/{MAX_CONNECTION_ERRORS})")
            
            if self.connection_errors >= MAX_CONNECTION_ERRORS:
                logging.critical("Max connection errors reached. Shutting down...")
                return False
        else:
            self.connection_errors = 0  # Reset on successful check
        
        return True


# =====================================
# JOURNALING (ENHANCED)
# =====================================

def init_journal():
    """Initialize trade journal CSV"""
    if not os.path.exists(JOURNAL_FILE):
        df = pd.DataFrame(columns=[
            'open_time', 'action', 'lot_size', 'entry_price', 'sl_price', 'tp_price',
            'close_time', 'exit_price', 'pnl', 'pnl_pct', 'exit_reason', 'ticket'
        ])
        df.to_csv(JOURNAL_FILE, index=False)
        logging.info("Trade journal initialized")


def log_trade(trade_data: dict):
    """
    Log trade to journal.
    
    Parameters:
        trade_data: dict with keys matching journal columns
    """
    try:
        df_new = pd.DataFrame([trade_data])
        df_new.to_csv(JOURNAL_FILE, mode='a', header=False, index=False)
        logging.info(f"Trade logged: {trade_data['ticket']}")
    except Exception as e:
        logging.error(f"Failed to log trade: {e}")


# =====================================
# MT5 HELPERS (ENHANCED)
# =====================================

def connect_mt5():
    """Connect to MT5 with error handling"""
    logging.info("Connecting to MetaTrader 5...")
    
    if not mt5.initialize():
        error = mt5.last_error()
        logging.error(f"MT5 connection failed: {error}")
        return False
    
    # Verify connection
    acc = mt5.account_info()
    if acc is None:
        logging.error("Connected but cannot get account info")
        return False
    
    logging.info(f"Connected successfully")
    logging.info(f"Account: {acc.login}, Server: {acc.server}")
    logging.info(f"Balance: ${acc.balance:.2f}, Equity: ${acc.equity:.2f}")
    
    return True


def get_candles(n=200):
    """
    Get historical candles from MT5.
    
    Parameters:
        n: number of candles
        
    Returns:
        pd.DataFrame or None
    """
    try:
        rates = mt5.copy_rates_from_pos(SYMBOL, TIMEFRAME, 0, n)
        if rates is None or len(rates) == 0:
            logging.error(f"Failed to get candles: {mt5.last_error()}")
            return None
        
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        df.set_index('time', inplace=True)
        
        return df
    
    except Exception as e:
        logging.error(f"Error getting candles: {e}")
        return None


def spread_ok():
    """
    Check if current spread is acceptable.
    
    Returns:
        bool: True if spread OK
    """
    try:
        sym = mt5.symbol_info(SYMBOL)
        tick = mt5.symbol_info_tick(SYMBOL)
        
        if not sym or not tick:
            return False
        
        current_spread = (tick.ask - tick.bid) / sym.point
        avg_spread = sym.spread
        
        is_ok = current_spread <= avg_spread * MAX_SPREAD_MULTIPLIER
        
        if not is_ok:
            logging.warning(f"Spread too high: {current_spread:.1f} (avg: {avg_spread:.1f})")
        
        return is_ok
    
    except Exception as e:
        logging.error(f"Error checking spread: {e}")
        return False


# =====================================
# RISK MANAGEMENT (FIXED)
# =====================================

def calculate_lot(entry, sl):
    """
    Calculate position size based on risk percentage.
    
    FIXED: Better error handling and margin checks
    
    Parameters:
        entry: entry price
        sl: stop loss price
        
    Returns:
        float: lot size
    """
    try:
        acc = mt5.account_info()
        sym = mt5.symbol_info(SYMBOL)
        
        if not acc or not sym:
            logging.error("Cannot get account/symbol info")
            return MIN_LOT
        
        if acc.balance <= 0:
            logging.error("Invalid balance")
            return MIN_LOT
        
        # Calculate risk amount
        risk_amt = acc.balance * (RISK_PERCENT / 100)
        
        # Calculate SL distance in points
        sl_distance = abs(entry - sl)
        sl_points = sl_distance / sym.point
        
        if sl_points == 0:
            logging.error("SL distance is zero")
            return MIN_LOT
        
        # Calculate lot size
        # risk_amt = lot_size * sl_points * tick_value
        lot = risk_amt / (sl_points * sym.trade_tick_value)
        
        # Round to volume step
        lot = round(lot / sym.volume_step) * sym.volume_step
        
        # Ensure within min/max bounds
        lot = max(sym.volume_min, min(lot, sym.volume_max))
        
        # Check margin requirements
        required_margin = lot * sym.margin_initial
        if acc.margin_free < required_margin:
            logging.warning(f"Insufficient margin. Required: ${required_margin:.2f}, Available: ${acc.margin_free:.2f}")
            return MIN_LOT
        
        logging.info(f"Calculated lot size: {lot:.2f} (Risk: ${risk_amt:.2f}, SL: {sl_points:.0f} points)")
        
        return lot
    
    except Exception as e:
        logging.error(f"Error calculating lot size: {e}")
        return MIN_LOT


# =====================================
# SIGNAL PREDICTION (FIXED)
# =====================================

def predict_signal(buy_model, sell_model, buy_threshold, sell_threshold):
    """
    Predict trading signal using ML models.
    
    FIXED: Use auto-detected features from DataFrame
    
    Parameters:
        buy_model, sell_model: trained models
        buy_threshold, sell_threshold: probability thresholds
        
    Returns:
        str or None: "BUY", "SELL", or None
    """
    try:
        # Get candles and add features
        df = get_candles(n=200)
        if df is None or len(df) < 100:
            logging.warning("Insufficient candle data")
            return None
        
        # Add features using SAME function as training
        df_features = add_all_features(df)
        
        if df_features is None or len(df_features) < 50:
            logging.warning("Insufficient data after adding features")
            return None
        
        # Validate features
        if not validate_features(df_features):
            logging.error("Feature validation failed")
            return None
        
        # Auto-detect feature columns from actual DataFrame
        feature_cols = get_feature_columns(df_features)
        
        # Get latest candle features
        X = df_features[feature_cols].iloc[-1:].values
        
        # Predict probabilities
        buy_proba = buy_model.predict_proba(X)[0][1]
        sell_proba = sell_model.predict_proba(X)[0][1]
        
        logging.info(f"Prediction - BUY: {buy_proba:.3f}, SELL: {sell_proba:.3f}")
        
        # Use optimal thresholds
        if buy_proba >= buy_threshold:
            logging.info(f"BUY signal detected (confidence: {buy_proba:.3f})")
            return "BUY"
        
        if sell_proba >= sell_threshold:
            logging.info(f"SELL signal detected (confidence: {sell_proba:.3f})")
            return "SELL"
        
        return None
    
    except Exception as e:
        logging.error(f"Error predicting signal: {e}")
        import traceback
        traceback.print_exc()
        return None


# =====================================
# ORDER EXECUTION (ENHANCED)
# =====================================

def place_order(signal, atr):
    """
    Place order based on signal.
    
    Parameters:
        signal: "BUY" or "SELL"
        atr: current ATR value
        
    Returns:
        int or None: order ticket or None
    """
    try:
        tick = mt5.symbol_info_tick(SYMBOL)
        sym = mt5.symbol_info(SYMBOL)
        
        if not tick or not sym:
            logging.error("Cannot get tick/symbol info")
            return None
        
        # Calculate entry, SL, TP
        if signal == "BUY":
            price = tick.ask
            sl = price - (atr * RISK_MULTIPLIER)
            tp = price + (atr * REWARD_MULTIPLIER)
            order_type = mt5.ORDER_TYPE_BUY
        else:  # SELL
            price = tick.bid
            sl = price + (atr * RISK_MULTIPLIER)
            tp = price - (atr * REWARD_MULTIPLIER)
            order_type = mt5.ORDER_TYPE_SELL
        
        # Calculate lot size
        lot = calculate_lot(price, sl)
        
        if lot < MIN_LOT:
            logging.error("Calculated lot size too small")
            return None
        
        # Prepare request
        request = {
            'action': mt5.TRADE_ACTION_DEAL,
            'symbol': SYMBOL,
            'volume': lot,
            'type': order_type,
            'price': price,
            'sl': sl,
            'tp': tp,
            'deviation': 20,  # Increased for better fill rate
            'magic': MAGIC_NUMBER,
            'comment': 'ML Bot',
            'type_filling': sym.filling_mode,
        }
        
        # Send order
        logging.info(f"Sending {signal} order: {lot:.2f} @ {price:.5f}, SL: {sl:.5f}, TP: {tp:.5f}")
        
        result = mt5.order_send(request)
        
        if result is None:
            logging.error("Order send returned None")
            return None
        
        if result.retcode == mt5.TRADE_RETCODE_DONE:
            logging.info(f"✅ Order executed successfully: Ticket {result.order}")
            
            # Log to journal (initial entry)
            log_trade({
                'open_time': datetime.now(TIMEZONE),
                'action': signal,
                'lot_size': lot,
                'entry_price': price,
                'sl_price': sl,
                'tp_price': tp,
                'close_time': None,
                'exit_price': None,
                'pnl': None,
                'pnl_pct': None,
                'exit_reason': None,
                'ticket': result.order
            })
            
            return result.order
        
        else:
            logging.error(f"Order failed: {result.retcode} - {result.comment}")
            return None
    
    except Exception as e:
        logging.error(f"Error placing order: {e}")
        import traceback
        traceback.print_exc()
        return None


# =====================================
# POSITION MONITORING (FIXED)
# =====================================

def monitor_and_close_positions(state):
    """
    Monitor all open positions and close if needed.
    
    CRITICAL FIXES:
    1. Loop ALL positions, not just first one
    2. Use pos.profit for PnL, not manual calculation
    3. Properly update state with actual PnL
    
    Parameters:
        state: DailyState object
    """
    try:
        positions = mt5.positions_get(symbol=SYMBOL, magic=MAGIC_NUMBER)
        
        if not positions:
            return  # No positions to monitor
        
        now = datetime.now(TIMEZONE)
        
        # CRITICAL FIX: Loop through ALL positions
        for pos in positions:
            
            # Get current price
            tick = mt5.symbol_info_tick(SYMBOL)
            if not tick:
                logging.error(f"Cannot get tick for position {pos.ticket}")
                continue
            
            current_price = tick.bid if pos.type == mt5.POSITION_TYPE_BUY else tick.ask
            
            # CRITICAL FIX: Use MT5's calculated profit
            pnl = pos.profit
            
            exit_reason = None
            
            # Check if TP/SL hit (MT5 should auto-close, but double check)
            if pos.type == mt5.POSITION_TYPE_BUY:
                if pos.sl > 0 and current_price <= pos.sl:
                    exit_reason = "SL"
                elif pos.tp > 0 and current_price >= pos.tp:
                    exit_reason = "TP"
            else:  # SELL
                if pos.sl > 0 and current_price >= pos.sl:
                    exit_reason = "SL"
                elif pos.tp > 0 and current_price <= pos.tp:
                    exit_reason = "TP"
            
            # Check midnight close
            if not exit_reason and now.time() >= MIDNIGHT_CLOSE_TIME:
                exit_reason = "MIDNIGHT_CLOSE"
            
            # Close position if needed
            if exit_reason:
                logging.info(f"Closing position {pos.ticket} - Reason: {exit_reason}, PnL: ${pnl:.2f}")
                
                # Close request
                close_request = {
                    'action': mt5.TRADE_ACTION_DEAL,
                    'symbol': pos.symbol,
                    'position': pos.ticket,
                    'volume': pos.volume,
                    'type': mt5.ORDER_TYPE_SELL if pos.type == mt5.POSITION_TYPE_BUY else mt5.ORDER_TYPE_BUY,
                    'price': current_price,
                    'deviation': 20,
                    'magic': MAGIC_NUMBER,
                    'comment': f'Bot close ({exit_reason})',
                    'type_filling': mt5.symbol_info(SYMBOL).filling_mode
                }
                
                result = mt5.order_send(close_request)
                
                if result and result.retcode == mt5.TRADE_RETCODE_DONE:
                    logging.info(f"✅ Position {pos.ticket} closed successfully")
                    
                    # CRITICAL FIX: Update state with ACTUAL PnL
                    state.update(pnl)
                    
                    # Update journal
                    pnl_pct = (pnl / mt5.account_info().balance) * 100
                    log_trade({
                        'open_time': datetime.fromtimestamp(pos.time, TIMEZONE),
                        'action': 'BUY' if pos.type == mt5.POSITION_TYPE_BUY else 'SELL',
                        'lot_size': pos.volume,
                        'entry_price': pos.price_open,
                        'sl_price': pos.sl,
                        'tp_price': pos.tp,
                        'close_time': now,
                        'exit_price': current_price,
                        'pnl': pnl,
                        'pnl_pct': pnl_pct,
                        'exit_reason': exit_reason,
                        'ticket': pos.ticket
                    })
                else:
                    logging.error(f"Failed to close position {pos.ticket}: {result.comment if result else 'Unknown error'}")
    
    except Exception as e:
        logging.error(f"Error monitoring positions: {e}")
        import traceback
        traceback.print_exc()


# =====================================
# MAIN LOOP (FIXED)
# =====================================

def main():
    """
    Main trading loop.
    """
    
    logging.info("="*60)
    logging.info("PRODUCTION-GRADE ML TRADING BOT STARTING")
    logging.info("="*60)
    
    # Connect to MT5
    if not connect_mt5():
        logging.critical("Cannot start bot - MT5 connection failed")
        return
    
    # Initialize journal
    init_journal()
    
    # Load models
    logging.info("Loading ML models...")
    try:
        buy_model = joblib.load(MODEL_BUY_FILE)
        sell_model = joblib.load(MODEL_SELL_FILE)
        logging.info("✅ Models loaded successfully")
        
        # Load metadata (includes optimal thresholds)
        if os.path.exists(MODEL_METADATA_FILE):
            metadata = joblib.load(MODEL_METADATA_FILE)
            buy_threshold = metadata.get('buy_threshold', DEFAULT_BUY_THRESHOLD)
            sell_threshold = metadata.get('sell_threshold', DEFAULT_SELL_THRESHOLD)
            logging.info(f"Using optimal thresholds - BUY: {buy_threshold:.3f}, SELL: {sell_threshold:.3f}")
        else:
            buy_threshold = DEFAULT_BUY_THRESHOLD
            sell_threshold = DEFAULT_SELL_THRESHOLD
            logging.warning(f"Using default thresholds - BUY: {buy_threshold:.3f}, SELL: {sell_threshold:.3f}")
        
    except Exception as e:
        logging.critical(f"Failed to load models: {e}")
        mt5.shutdown()
        return
    
    # Initialize state
    state = DailyState()
    
    logging.info("="*60)
    logging.info("BOT RUNNING - Press Ctrl+C to stop")
    logging.info("="*60)
    
    # Main loop
    loop_count = 0
    
    while True:
        try:
            loop_count += 1
            
            # Check if new day
            now = datetime.now(TIMEZONE)
            if now.date() != state.date:
                logging.info(f"New day detected: {now.date()}")
                state.reset()
            
            # CRITICAL: Monitor positions at START of every loop
            monitor_and_close_positions(state)
            
            # Check connection health
            if not state.check_connection_health():
                logging.critical("Connection health check failed. Stopping bot.")
                break
            
            # Check if in cooldown
            if state.cooldown_active():
                dt.sleep(60)
                continue
            
            # Check daily limits
            can_trade, reason = state.check_daily_limits()
            if not can_trade:
                logging.warning(f"Trading stopped: {reason}")
                dt.sleep(300)  # Wait 5 minutes before checking again
                continue
            
            # Check trading session
            if not (TRADING_SESSION_START <= now.time() <= TRADING_SESSION_END):
                if loop_count % 12 == 0:  # Log every ~1 hour
                    logging.info(f"Outside trading hours ({TRADING_SESSION_START} - {TRADING_SESSION_END})")
                dt.sleep(300)
                continue
            
            # Check if already have open position
            open_positions = mt5.positions_get(symbol=SYMBOL, magic=MAGIC_NUMBER)
            if open_positions:
                if loop_count % 6 == 0:  # Log every ~30 minutes
                    logging.info(f"{len(open_positions)} position(s) already open")
                dt.sleep(300)
                continue
            
            # Check spread
            if not spread_ok():
                dt.sleep(60)
                continue
            
            # Get trading signal
            signal = predict_signal(buy_model, sell_model, buy_threshold, sell_threshold)
            
            if not signal:
                dt.sleep(60)  # Check again in 1 minute
                continue
            
            # Get current ATR for position sizing
            df_atr = add_all_features(get_candles())
            if df_atr is None:
                logging.error("Cannot get ATR")
                dt.sleep(60)
                continue
            
            atr = df_atr[f'ATRr_{ATR_PERIOD}'].iloc[-1]
            
            if pd.isna(atr) or atr == 0:
                logging.error("Invalid ATR value")
                dt.sleep(60)
                continue
            
            # Place order
            ticket = place_order(signal, atr)
            
            if ticket:
                logging.info(f"🎯 New trade opened: {signal} - Ticket {ticket}")
            
            # Wait before next loop
            dt.sleep(30)  # Check every 30 seconds
        
        except KeyboardInterrupt:
            logging.info("Bot stopped by user")
            break
        
        except Exception as e:
            logging.error(f"Error in main loop: {e}")
            import traceback
            traceback.print_exc()
            dt.sleep(60)
    
    # Cleanup
    logging.info("Closing all open positions...")
    
    positions = mt5.positions_get(symbol=SYMBOL, magic=MAGIC_NUMBER)
    if positions:
        for pos in positions:
            tick = mt5.symbol_info_tick(SYMBOL)
            current_price = tick.bid if pos.type == mt5.POSITION_TYPE_BUY else tick.ask
            
            close_request = {
                'action': mt5.TRADE_ACTION_DEAL,
                'symbol': pos.symbol,
                'position': pos.ticket,
                'volume': pos.volume,
                'type': mt5.ORDER_TYPE_SELL if pos.type == mt5.POSITION_TYPE_BUY else mt5.ORDER_TYPE_BUY,
                'price': current_price,
                'deviation': 20,
                'magic': MAGIC_NUMBER,
                'comment': 'Bot shutdown',
                'type_filling': mt5.symbol_info(SYMBOL).filling_mode
            }
            
            mt5.order_send(close_request)
    
    mt5.shutdown()
    logging.info("Bot shutdown complete")


if __name__ == '__main__':
    main()
