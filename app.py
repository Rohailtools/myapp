import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle
import requests
import io
import random
from datetime import datetime, timedelta

# Try to import sklearn - if not available, ML features will be disabled
try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

import warnings
warnings.filterwarnings('ignore')

# ==================== PAGE CONFIG ====================
st.set_page_config(
    page_title="RJ Algo Tools PRO",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== CUSTOM DARK THEME CSS ====================
st.markdown("""
<style>
    .main { background-color: #0e1117; }
    .stMetric {
        background-color: #1e1e2e;
        border-radius: 10px;
        padding: 15px;
        border-left: 4px solid #00d4ff;
    }
    .stTabs [data-baseweb="tab-list"] { gap: 10px; }
    .stTabs [data-baseweb="tab"] {
        background-color: #1e1e2e;
        border-radius: 8px 8px 0 0;
        padding: 10px 20px;
        color: #ffffff;
        font-weight: 500;
    }
    .stTabs [aria-selected="true"] {
        background-color: #00d4ff !important;
        color: #000000 !important;
        font-weight: bold;
    }
    .stButton>button {
        background: linear-gradient(90deg, #00d4ff, #0099cc);
        color: black;
        font-weight: bold;
        border-radius: 10px;
        padding: 12px 30px;
        font-size: 16px;
    }
    .stTextArea textarea {
        background-color: #1e1e2e;
        color: #00ff88;
        font-family: 'Courier New', monospace;
        border: 1px solid #00d4ff;
        font-size: 13px;
    }
    h1, h2, h3 { color: #00d4ff !important; }
    .css-1d391kg { background-color: #161b22; }
</style>
""", unsafe_allow_html=True)

# ==================== API KEYS ====================
ALPHA_VANTAGE_KEY = "0XGOOAZ1PBFYIUFI"
TELEGRAM_BOT_TOKEN = st.secrets.get("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = st.secrets.get("TELEGRAM_CHAT_ID", "")

# ==================== SIDEBAR ====================
st.sidebar.markdown("<h1 style='color:#00d4ff; text-align:center;'>🚀 RJ Algo PRO</h1>", unsafe_allow_html=True)
st.sidebar.markdown("<p style='text-align:center; color:#888;'>v2.0 Institutional Grade</p>", unsafe_allow_html=True)
st.sidebar.markdown("---")

st.sidebar.header("⚙️ Global Settings")
PAIR = st.sidebar.selectbox("📊 Trading Pair", ['EURUSD', 'GBPUSD', 'USDJPY', 'XAUUSD', 'BTCUSD'])
TIMEFRAME = st.sidebar.selectbox("⏱️ Timeframe", ['1m', '5m', '15m', '1h', '4h', '1d'])
LOT_SIZE = st.sidebar.number_input("💰 Lot Size", min_value=0.01, value=0.10, step=0.01)
LEVERAGE = st.sidebar.number_input("⚡ Leverage", min_value=1, max_value=2000, value=100)

st.sidebar.markdown("---")
st.sidebar.header("🛡️ Risk Settings")
SL_PIPS = st.sidebar.number_input("🛑 Stop Loss (Pips)", value=300)
TP_PIPS = st.sidebar.number_input("🎯 Take Profit (Pips)", value=600)
USE_SPREAD = st.sidebar.checkbox("📉 Apply Real Spread", value=True)
USE_COMMISSION = st.sidebar.checkbox("💸 Apply Commission", value=True)
COMMISSION_PER_LOT = st.sidebar.number_input("Commission $/Lot", value=7.0)

st.sidebar.markdown("---")
st.sidebar.header("🤖 Advanced Features")
ENABLE_ML = st.sidebar.checkbox("🧠 AI Prediction", value=False)
ENABLE_MC = st.sidebar.checkbox("🎲 Monte Carlo Sim", value=True)
ENABLE_WALKFORWARD = st.sidebar.checkbox("🚶 Walk-Forward", value=False)
ENABLE_OPTIMIZER = st.sidebar.checkbox("⚡ Auto Optimizer", value=False)
ENABLE_TELEGRAM = st.sidebar.checkbox("📱 Telegram Alerts", value=False)

# ==================== STRATEGY TEMPLATES ====================
STRATEGY_TEMPLATES = {
    "Liquidity Sweep (Default)": """def my_strategy(df):
    df['Signal'] = 0
    df['EMA50'] = df['Close'].ewm(span=50, adjust=False).mean()
    df['Liq_High'] = df['High'].rolling(window=20).max().shift(1)
    df['Liq_Low'] = df['Low'].rolling(window=20).min().shift(1)
    df['Bull'] = df['Close'] > df['Open']
    df['Bear'] = df['Open'] > df['Close']
    df.loc[(df['Close'] > df['EMA50']) & (df['Low'] < df['Liq_Low']) & (df['Bull']), 'Signal'] = 1
    df.loc[(df['Close'] < df['EMA50']) & (df['High'] > df['Liq_High']) & (df['Bear']), 'Signal'] = -1
    return df""",

    "RSI Mean Reversion": """def my_strategy(df):
    df['Signal'] = 0
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    df['BB_Mid'] = df['Close'].rolling(20).mean()
    std = df['Close'].rolling(20).std()
    df['BB_Lower'] = df['BB_Mid'] - 2 * std
    df['BB_Upper'] = df['BB_Mid'] + 2 * std
    df.loc[(df['Close'] < df['BB_Lower']) & (df['RSI'] < 25), 'Signal'] = 1
    df.loc[(df['Close'] > df['BB_Upper']) & (df['RSI'] > 75), 'Signal'] = -1
    return df""",

    "EMA Crossover": """def my_strategy(df):
    df['Signal'] = 0
    df['EMA9'] = df['Close'].ewm(span=9, adjust=False).mean()
    df['EMA21'] = df['Close'].ewm(span=21, adjust=False).mean()
    df['EMA50'] = df['Close'].ewm(span=50, adjust=False).mean()
    df.loc[(df['EMA9'] > df['EMA21']) & (df['EMA21'] > df['EMA50']), 'Signal'] = 1
    df.loc[(df['EMA9'] < df['EMA21']) & (df['EMA21'] < df['EMA50']), 'Signal'] = -1
    return df""",

    "Breakout Strategy": """def my_strategy(df):
    df['Signal'] = 0
    df['Resistance'] = df['High'].rolling(window=20).max().shift(1)
    df['Support'] = df['Low'].rolling(window=20).min().shift(1)
    df.loc[df['Close'] > df['Resistance'], 'Signal'] = 1
    df.loc[df['Close'] < df['Support'], 'Signal'] = -1
    return df""",

    "MACD Strategy": """def my_strategy(df):
    df['Signal'] = 0
    ema12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = ema12 - ema26
    df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['Histogram'] = df['MACD'] - df['Signal_Line']
    df.loc[(df['MACD'] > df['Signal_Line']) & (df['MACD'].shift(1) < df['Signal_Line'].shift(1)), 'Signal'] = 1
    df.loc[(df['MACD'] < df['Signal_Line']) & (df['MACD'].shift(1) > df['Signal_Line'].shift(1)), 'Signal'] = -1
    return df""",

    "Supply & Demand Zones": """def my_strategy(df):
    df['Signal'] = 0
    df['Pivot_High'] = df['High'].rolling(window=5, center=True).max() == df['High']
    df['Pivot_Low'] = df['Low'].rolling(window=5, center=True).min() == df['Low']
    df['Supply'] = df['High'].where(df['Pivot_High']).rolling(20).max()
    df['Demand'] = df['Low'].where(df['Pivot_Low']).rolling(20).min()
    df['Supply'] = df['Supply'].ffill()
    df['Demand'] = df['Demand'].ffill()
    df.loc[(df['Close'] > df['Supply'].shift(1)) & (df['Close'].shift(1) < df['Supply'].shift(2)), 'Signal'] = 1
    df.loc[(df['Close'] < df['Demand'].shift(1)) & (df['Close'].shift(1) > df['Demand'].shift(2)), 'Signal'] = -1
    return df""",

    "Custom Code": """def my_strategy(df):
    df['Signal'] = 0
    # Write your strategy here
    # df['Signal'] = 1 for BUY, -1 for SELL, 0 for no trade
    return df"""
}

# ==================== DATA ENGINE ====================
@st.cache_data(ttl=1800)
def load_data(pair, timeframe):
    period_map = {'1m': '7d', '5m': '60d', '15m': '60d', '1h': '730d', '4h': '730d', '1d': '5y'}
    period = period_map.get(timeframe, '1y')
    
    if pair in ['EURUSD', 'GBPUSD', 'USDJPY'] and ALPHA_VANTAGE_KEY:
        try:
            from_sym, to_sym = pair[:3], pair[3:]
            interval_map = {'1m': '1min', '5m': '5min', '15m': '15min', '1h': '60min', '4h': '60min', '1d': 'daily'}
            av_interval = interval_map.get(timeframe, '60min')
            url = f"https://www.alphavantage.co/query?function=FX_INTRADAY&from_symbol={from_sym}&to_symbol={to_sym}&interval={av_interval}&outputsize=full&apikey={ALPHA_VANTAGE_KEY}"
            r = requests.get(url, timeout=15)
            data = r.json()
            ts_key = 'Time Series FX (' + av_interval + ')'
            if ts_key in data:
                df = pd.DataFrame.from_dict(data[ts_key], orient='index')
                df = df.rename(columns={'1. open': 'Open', '2. high': 'High', '3. low': 'Low', '4. close': 'Close'})
                df.index = pd.to_datetime(df.index)
                df = df.sort_index()
                for col in ['Open', 'High', 'Low', 'Close']:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                df = df.dropna()
                if len(df) > 50:
                    return df, "Alpha Vantage"
        except:
            pass
    
    ticker_map = {'EURUSD': 'EURUSD=X', 'GBPUSD': 'GBPUSD=X', 'USDJPY': 'USDJPY=X', 'XAUUSD': 'GC=F', 'BTCUSD': 'BTC-USD'}
    try:
        df = yf.download(ticker_map.get(pair, pair), period=period, interval=timeframe, progress=False, auto_adjust=False, threads=False)
        if df.empty:
            return None, "Empty data"
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        col_map = {}
        for c in df.columns:
            s = str(c).upper()
            if 'OPEN' in s and 'ADJ' not in s:
                col_map[c] = 'Open'
            elif 'HIGH' in s:
                col_map[c] = 'High'
            elif 'LOW' in s:
                col_map[c] = 'Low'
            elif 'CLOSE' in s and 'ADJ' not in s:
                col_map[c] = 'Close'
        if col_map:
            df = df.rename(columns=col_map)
        df = df[['Open', 'High', 'Low', 'Close']].copy().dropna()
        for c in ['Open', 'High', 'Low', 'Close']:
            df[c] = pd.to_numeric(df[c], errors='coerce')
        df = df.dropna()
        return (df, "Yahoo Finance") if len(df) > 50 else (None, "Too few rows")
    except Exception as e:
        return None, str(e)

# ==================== COSTS & MULTIPLIERS ====================
SPREAD_MAP = {'EURUSD': 1.0, 'GBPUSD': 1.5, 'USDJPY': 1.2, 'XAUUSD': 20.0, 'BTCUSD': 50.0}

def apply_costs(pnl, pair, lot_size, use_spread=True, use_commission=True):
    costs = 0
    if use_spread and pair in SPREAD_MAP:
        if pair in ['EURUSD', 'GBPUSD']:
            spread_cost = (SPREAD_MAP[pair] * 0.0001) * lot_size * 0.1 * 100000
        elif pair == 'USDJPY':
            spread_cost = (SPREAD_MAP[pair] * 0.01) * lot_size * 0.1 * 100000
        else:
            spread_cost = (SPREAD_MAP[pair] * 0.10) * lot_size * 1.0
        costs += spread_cost
    if use_commission:
        costs += COMMISSION_PER_LOT * lot_size
    return pnl - costs

def get_multiplier(pair):
    if pair in ['EURUSD', 'GBPUSD']:
        return 0.0001
    elif pair == 'USDJPY':
        return 0.01
    else:
        return 0.10

# ==================== INTRA-CANDLE SIMULATION ====================
def simulate_intracandle(entry, sl, tp, high, low, trade_type):
    if trade_type == 1:
        if low <= sl:
            return sl, "SL Hit"
        elif high >= tp:
            return tp, "TP Hit"
    else:
        if high >= sl:
            return sl, "SL Hit"
        elif low <= tp:
            return tp, "TP Hit"
    return 0, ""

# ==================== BACKTEST ENGINE ====================
def run_backtest(df, pair, lot_size, sl_pips, tp_pips, use_spread=True, use_commission=True):
    mult = get_multiplier(pair)
    pip_val = 0.1 if pair in ['EURUSD', 'GBPUSD', 'USDJPY'] else 1.0
    trades = []
    in_trade = False
    entry_p = sl_p = tp_p = 0
    trade_type = 0
    
    for i in range(1, len(df)):
        if not in_trade:
            if df['Signal'].iloc[i] != 0:
                in_trade = True
                trade_type = int(df['Signal'].iloc[i])
                entry_p = float(df['Open'].iloc[i])
                sl_p = entry_p - (sl_pips * mult * trade_type)
                tp_p = entry_p + (tp_pips * mult * trade_type)
        else:
            exit_p, reason = simulate_intracandle(
                entry_p, sl_p, tp_p,
                float(df['High'].iloc[i]),
                float(df['Low'].iloc[i]), trade_type
            )
            if exit_p == 0:
                if trade_type == 1 and df['Signal'].iloc[i] == -1:
                    exit_p = float(df['Open'].iloc[i])
                    reason = "Signal Exit"
                elif trade_type == -1 and df['Signal'].iloc[i] == 1:
                    exit_p = float(df['Open'].iloc[i])
                    reason = "Signal Exit"
            if exit_p != 0:
                in_trade = False
                pnl_pts = ((exit_p - entry_p) * trade_type) / mult
                pnl_dol = pnl_pts * pip_val * lot_size
                pnl_dol = apply_costs(pnl_dol, pair, lot_size, use_spread, use_commission)
                trades.append({
                    'Type': 'BUY' if trade_type == 1 else 'SELL',
                    'Entry': round(entry_p, 4),
                    'Exit': round(exit_p, 4),
                    'Pnl': round(pnl_dol, 2),
                    'Reason': reason,
                    'RRR': f"1:{tp_pips/sl_pips:.1f}" if reason == "TP Hit" else "1:0",
                    'Date': df.index[i],
                    'EntryIdx': i,
                    'SL': sl_p,
                    'TP': tp_p
                })
    return trades

# ==================== ANALYTICS ====================
def calculate_metrics(trades, initial_equity=10000):
    if not trades:
        return {}
    pnls = [t['Pnl'] for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p < 0]
    equity = [initial_equity]
    for p in pnls:
        equity.append(equity[-1] + p)
    peak = initial_equity
    max_dd = 0
    dd_series = []
    for e in equity:
        if e > peak:
            peak = e
        dd = (peak - e) / peak * 100
        dd_series.append(dd)
        if dd > max_dd:
            max_dd = dd
    returns = np.array(pnls)
    avg_ret = np.mean(returns)
    std_ret = np.std(returns)
    sharpe = (avg_ret / std_ret * np.sqrt(252)) if std_ret > 0 else 0
    downside = np.array([p for p in pnls if p < 0])
    downside_std = np.std(downside) if len(downside) > 0 else 0
    sortino = (avg_ret / downside_std * np.sqrt(252)) if downside_std > 0 else 0
    max_cw = max_cl = cw = cl = 0
    for p in pnls:
        if p > 0:
            cw += 1
            cl = 0
            max_cw = max(max_cw, cw)
        else:
            cl += 1
            cw = 0
            max_cl = max(max_cl, cl)
    return {
        'total_trades': len(trades),
        'win_rate': (len(wins) / len(trades)) * 100,
        'profit_factor': abs(sum(wins) / sum(losses)) if losses and sum(losses) != 0 else float('inf'),
        'avg_win': np.mean(wins) if wins else 0,
        'avg_loss': np.mean(losses) if losses else 0,
        'net_pnl': sum(pnls),
        'max_dd': max_dd,
        'sharpe': sharpe,
        'sortino': sortino,
        'equity': equity,
        'dd_series': dd_series,
        'max_consec_wins': max_cw,
        'max_consec_losses': max_cl,
        'total_wins': len(wins),
        'total_losses': len(losses)
    }

# ==================== MONTE CARLO ====================
def monte_carlo_simulation(trades, n_sims=1000, initial_equity=10000):
    if not trades:
        return None
    pnls = [t['Pnl'] for t in trades]
    final_eqs = []
    max_dds = []
    ruin_count = 0
    for _ in range(n_sims):
        random.shuffle(pnls)
        eq = [initial_equity]
        peak = initial_equity
        max_dd = 0
        for p in pnls:
            eq.append(eq[-1] + p)
            if eq[-1] > peak:
                peak = eq[-1]
            dd = (peak - eq[-1]) / peak * 100
            if dd > max_dd:
                max_dd = dd
            if eq[-1] <= initial_equity * 0.5:
                break
        final_eqs.append(eq[-1])
        max_dds.append(max_dd)
        if eq[-1] <= initial_equity * 0.5:
            ruin_count += 1
    return {
        'best': max(final_eqs),
        'worst': min(final_eqs),
        'avg': np.mean(final_eqs),
        'median': np.median(final_eqs),
        'ruin_prob': (ruin_count / n_sims) * 100,
        'avg_max_dd': np.mean(max_dds)
    }

# ==================== WALK FORWARD ====================
def walk_forward_analysis(df, pair, lot_size, sl_pips, tp_pips, strategy_func, n_splits=3):
    split_size = len(df) // n_splits
    results = []
    for i in range(n_splits - 1):
        test_df = df.iloc[split_size * (i + 1):split_size * (i + 2)].copy()
        test_df = strategy_func(test_df)
        trades = run_backtest(test_df, pair, lot_size, sl_pips, tp_pips, False, False)
        m = calculate_metrics(trades)
        results.append({
            'Period': f"Split {i+1}",
            'Trades': m.get('total_trades', 0),
            'WinRate': f"{m.get('win_rate', 0):.1f}%",
            'NetPnL': f"${m.get('net_pnl', 0):.2f}",
            'Sharpe': f"{m.get('sharpe', 0):.2f}",
            'MaxDD': f"{m.get('max_dd', 0):.1f}%"
        })
    return pd.DataFrame(results)

# ==================== AUTO OPTIMIZER ====================
def optimize_strategy(df, pair, lot_size, strategy_func):
    best_pnl = -999999
    best_params = None
    results = []
    sl_range = range(50, 501, 50)
    tp_range = range(100, 1001, 50)
    total = len(sl_range) * len(tp_range)
    count = 0
    progress_bar = st.progress(0)
    status = st.empty()
    for sl in sl_range:
        for tp in tp_range:
            count += 1
            progress_bar.progress(count / total)
            status.text(f"Testing SL={sl}, TP={tp}... ({count}/{total})")
            df_temp = strategy_func(df.copy())
            trades = run_backtest(df_temp, pair, lot_size, sl, tp, False, False)
            if trades:
                pnl = sum(t['Pnl'] for t in trades)
                wr = len([t for t in trades if t['Pnl'] > 0]) / len(trades) * 100
                results.append({'SL': sl, 'TP': tp, 'PnL': pnl, 'WinRate': wr, 'Trades': len(trades)})
                if pnl > best_pnl:
                    best_pnl = pnl
                    best_params = (sl, tp)
    progress_bar.empty()
    status.empty()
    return best_params, best_pnl, pd.DataFrame(results)

# ==================== ML PREDICTOR ====================
def ml_predictor(df, pair):
    if not SKLEARN_AVAILABLE:
        return None, "scikit-learn not installed. Add 'scikit-learn' to requirements.txt"
    df_ml = df.copy()
    df_ml['Returns'] = df_ml['Close'].pct_change()
    df_ml['Volatility'] = df_ml['Returns'].rolling(20).std()
    df_ml['Momentum'] = df_ml['Close'] - df_ml['Close'].shift(10)
    delta = df_ml['Close'].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    df_ml['RSI'] = 100 - (100 / (1 + rs))
    df_ml['MA_Dist'] = (df_ml['Close'] - df_ml['Close'].rolling(20).mean()) / df_ml['Close']
    df_ml['BB_Width'] = (df_ml['Close'].rolling(20).std() * 2) / df_ml['Close'].rolling(20).mean()
    df_ml['Target'] = (df_ml['Close'].shift(-1) > df_ml['Close']).astype(int)
    features = ['Returns', 'Volatility', 'Momentum', 'RSI', 'MA_Dist', 'BB_Width']
    df_ml = df_ml.dropna()
    if len(df_ml) < 100:
        return None, "Not enough data for ML (need 100+ rows)"
    X = df_ml[features]
    y = df_ml['Target']
    split = int(len(df_ml) * 0.8)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]
    model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred) * 100
    importance = pd.DataFrame({
        'Feature': features,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False)
    latest = X.iloc[-1:].values
    pred_prob = model.predict_proba(latest)[0]
    pred = model.predict(latest)[0]
    return {
        'accuracy': accuracy,
        'prediction': 'UP' if pred == 1 else 'DOWN',
        'confidence': max(pred_prob) * 100,
        'importance': importance,
        'test_size': len(y_test)
    }, None

# ==================== TELEGRAM ====================
def send_telegram(message):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        return False
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        requests.post(url, json={"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "HTML"}, timeout=5)
        return True
    except:
        return False

# ==================== CANDLESTICK CHART ====================
def plot_strategy_chart(df, trades, pair, strategy_name, show_indicators=True):
    fig, axes = plt.subplots(3, 1, figsize=(16, 12), gridspec_kw={'height_ratios': [3, 1, 1]})
    fig.patch.set_facecolor('#0e1117')
    
    if len(df) > 300:
        df_plot = df.iloc[-300:].copy()
        start_idx = len(df) - 300
    else:
        df_plot = df.copy()
        start_idx = 0
    
    x_range = range(len(df_plot))
    dates = df_plot.index
    
    # MAIN CHART
    ax1 = axes[0]
    ax1.set_facecolor('#1e1e2e')
    
    for i, (idx, row) in enumerate(df_plot.iterrows()):
        open_p, high_p, low_p, close_p = row['Open'], row['High'], row['Low'], row['Close']
        color = '#00ff88' if close_p >= open_p else '#ff4444'
        height = abs(close_p - open_p)
        bottom = min(open_p, close_p)
        rect = Rectangle((i - 0.4, bottom), 0.8, height if height > 0 else 0.0001,
                         facecolor=color, edgecolor=color, linewidth=0.5, alpha=0.8)
        ax1.add_patch(rect)
        ax1.plot([i, i], [low_p, high_p], color='white', linewidth=0.5, alpha=0.7)
    
    if show_indicators:
        if 'EMA50' in df_plot.columns:
            ax1.plot(x_range, df_plot['EMA50'], color='#00d4ff', linewidth=1.5, label='EMA 50', alpha=0.9)
        if 'EMA9' in df_plot.columns:
            ax1.plot(x_range, df_plot['EMA9'], color='#ffaa00', linewidth=1.2, label='EMA 9', alpha=0.9)
        if 'EMA21' in df_plot.columns:
            ax1.plot(x_range, df_plot['EMA21'], color='#ff00ff', linewidth=1.2, label='EMA 21', alpha=0.9)
        if 'BB_Upper' in df_plot.columns:
            ax1.plot(x_range, df_plot['BB_Upper'], color='gray', linewidth=0.8, linestyle='--', alpha=0.5)
            ax1.plot(x_range, df_plot['BB_Lower'], color='gray', linewidth=0.8, linestyle='--', alpha=0.5)
            ax1.fill_between(x_range, df_plot['BB_Upper'], df_plot['BB_Lower'], alpha=0.05, color='gray')
        if 'Liq_High' in df_plot.columns:
            ax1.plot(x_range, df_plot['Liq_High'], color='#ff8800', linewidth=0.8, linestyle=':', alpha=0.6, label='Liq High')
            ax1.plot(x_range, df_plot['Liq_Low'], color='#ff8800', linewidth=0.8, linestyle=':', alpha=0.6, label='Liq Low')
        if 'Supply' in df_plot.columns:
            ax1.plot(x_range, df_plot['Supply'], color='#ff0000', linewidth=1, linestyle='--', alpha=0.5, label='Supply')
            ax1.plot(x_range, df_plot['Demand'], color='#00ff00', linewidth=1, linestyle='--', alpha=0.5, label='Demand')
    
    for trade in trades:
        entry_idx = trade.get('EntryIdx', 0) - start_idx
        if 0 <= entry_idx < len(df_plot):
            color = '#00ff88' if trade['Type'] == 'BUY' else '#ff4444'
            marker = '^' if trade['Type'] == 'BUY' else 'v'
            ax1.scatter(entry_idx, trade['Entry'], marker=marker, s=200, c=color,
                       edgecolors='white', linewidth=1.5, zorder=5, alpha=0.9)
            if 'SL' in trade and 'TP' in trade:
                ax1.hlines(trade['SL'], entry_idx - 2, entry_idx + 10, colors='#ff4444',
                          linewidth=1, linestyle='--', alpha=0.6)
                ax1.hlines(trade['TP'], entry_idx - 2, entry_idx + 10, colors='#00ff88',
                          linewidth=1, linestyle='--', alpha=0.6)
                ax1.text(entry_idx + 5, trade['SL'], 'SL', fontsize=7, color='#ff4444', alpha=0.8)
                ax1.text(entry_idx + 5, trade['TP'], 'TP', fontsize=7, color='#00ff88', alpha=0.8)
            pnl_color = '#00ff88' if trade['Pnl'] > 0 else '#ff4444'
            ax1.annotate(f"${trade['Pnl']}", xy=(entry_idx, trade['Entry']),
                        xytext=(entry_idx + 3, trade['Entry'] + (0.001 if trade['Type'] == 'BUY' else -0.001)),
                        fontsize=8, color=pnl_color, fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.2', facecolor='#1e1e2e', edgecolor=pnl_color, alpha=0.8))
    
    if 'Signal' in df_plot.columns:
        buy_signals = df_plot[df_plot['Signal'] == 1].index
        sell_signals = df_plot[df_plot['Signal'] == -1].index
        for idx in buy_signals:
            i = df_plot.index.get_loc(idx)
            ax1.scatter(i, df_plot.loc[idx, 'Low'] * 0.9995, marker='^', s=80, c='#00ff88',
                       alpha=0.4, edgecolors='none', zorder=4)
        for idx in sell_signals:
            i = df_plot.index.get_loc(idx)
            ax1.scatter(i, df_plot.loc[idx, 'High'] * 1.0005, marker='v', s=80, c='#ff4444',
                       alpha=0.4, edgecolors='none', zorder=4)
    
    ax1.set_title(f'{pair} | {strategy_name} | Live Strategy Chart', color='white', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Price', color='white')
    ax1.tick_params(colors='white')
    ax1.grid(True, alpha=0.2, color='gray')
    ax1.legend(loc='upper left', facecolor='#1e1e2e', edgecolor='white', labelcolor='white')
    ax1.set_xlim(-1, len(df_plot))
    n_ticks = min(10, len(dates))
    tick_positions = np.linspace(0, len(dates) - 1, n_ticks, dtype=int)
    ax1.set_xticks(tick_positions)
    ax1.set_xticklabels([dates[i].strftime('%m-%d %H:%M') for i in tick_positions], rotation=45, color='white')
    
    # SIGNAL PANEL
    ax2 = axes[1]
    ax2.set_facecolor('#1e1e2e')
    if 'Signal' in df_plot.columns:
        signal_colors = ['#00ff88' if s == 1 else '#ff4444' if s == -1 else '#333333' for s in df_plot['Signal']]
        ax2.bar(x_range, [1 if s != 0 else 0 for s in df_plot['Signal']], color=signal_colors, alpha=0.7, width=0.8)
        ax2.set_ylabel('Signal', color='white')
        ax2.set_ylim(-0.5, 1.5)
        ax2.set_yticks([0, 1])
        ax2.set_yticklabels(['None', 'Active'], color='white')
    else:
        ax2.text(0.5, 0.5, 'No Signal Data', ha='center', va='center', color='white', transform=ax2.transAxes)
    ax2.tick_params(colors='white')
    ax2.grid(True, alpha=0.2, color='gray')
    ax2.set_xticks(tick_positions)
    ax2.set_xticklabels([dates[i].strftime('%m-%d') for i in tick_positions], rotation=45, color='white')
    
    # INDICATOR PANEL
    ax3 = axes[2]
    ax3.set_facecolor('#1e1e2e')
    if 'RSI' in df_plot.columns:
        ax3.plot(x_range, df_plot['RSI'], color='#00d4ff', linewidth=1.2, label='RSI')
        ax3.axhline(y=70, color='#ff4444', linestyle='--', alpha=0.5, linewidth=0.8)
        ax3.axhline(y=30, color='#00ff88', linestyle='--', alpha=0.5, linewidth=0.8)
        ax3.fill_between(x_range, 30, 70, alpha=0.05, color='gray')
        ax3.set_ylabel('RSI', color='white')
        ax3.set_ylim(0, 100)
    elif 'MACD' in df_plot.columns:
        ax3.plot(x_range, df_plot['MACD'], color='#00d4ff', linewidth=1.2, label='MACD')
        ax3.plot(x_range, df_plot['Signal_Line'], color='#ffaa00', linewidth=1, label='Signal')
        hist_colors = ['#00ff88' if h > 0 else '#ff4444' for h in df_plot['Histogram']]
        ax3.bar(x_range, df_plot['Histogram'], color=hist_colors, alpha=0.5, width=0.8)
        ax3.axhline(y=0, color='white', linewidth=0.5, alpha=0.5)
        ax3.set_ylabel('MACD', color='white')
    else:
        ax3.text(0.5, 0.5, 'Add RSI or MACD to strategy for indicator panel',
                ha='center', va='center', color='white', transform=ax3.transAxes)
    ax3.tick_params(colors='white')
    ax3.grid(True, alpha=0.2, color='gray')
    ax3.set_xticks(tick_positions)
    ax3.set_xticklabels([dates[i].strftime('%m-%d') for i in tick_positions], rotation=45, color='white')
    ax3.legend(loc='upper left', facecolor='#1e1e2e', edgecolor='white', labelcolor='white')
    
    plt.tight_layout()
    return fig

# ==================== MAIN UI ====================
st.markdown("<h1 style='text-align:center; color:#00d4ff;'>🚀 RJ Algo Tools PRO v2.0</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; color:#888;'>Institutional Grade Backtesting with Real-Time Strategy Visualization</p>", unsafe_allow_html=True)
st.markdown("---")

tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "📊 Strategy & Backtest", "📈 Live Chart", "🎲 Monte Carlo", "🚶 Walk Forward",
    "⚡ Optimizer", "🧠 AI Predictor", "📱 Settings"
])

# ==================== TAB 1: STRATEGY & BACKTEST ====================
with tab1:
    st.header("🎯 Strategy Configuration")
    
    col1, col2 = st.columns([1, 2])
    with col1:
        st.subheader("Select Template")
        selected_template = st.selectbox("Choose Strategy", list(STRATEGY_TEMPLATES.keys()))
        st.caption("Template select karo ya Custom Code edit karo")
        
        if st.button("🔄 Reset to Template"):
            st.session_state['strategy_code'] = STRATEGY_TEMPLATES[selected_template]
            st.rerun()
    
    with col2:
        st.subheader("Strategy Code Editor")
        strategy_code = st.text_area(
            "Edit Python Code:",
            value=STRATEGY_TEMPLATES[selected_template],
            height=280,
            key='strategy_code'
        )
    
    st.markdown("---")
    run_btn = st.button("🚀 RUN PRO BACKTEST", use_container_width=True, type="primary")
    
    if run_btn:
        with st.spinner(f'Loading {PAIR} data...'):
            df, source = load_data(PAIR, TIMEFRAME)
        
        if df is None:
            st.error(f"❌ Data load failed: {source}")
        else:
            st.success(f"✅ Loaded {len(df)} rows from {source}")
            
            with st.expander("🔍 Data Preview"):
                st.dataframe(df.tail(10), use_container_width=True)
            
            try:
                exec(strategy_code, globals())
                df = my_strategy(df)
                
                if 'Signal' not in df.columns:
                    st.error("❌ Strategy must return 'Signal' column! (1=BUY, -1=SELL, 0=HOLD)")
                else:
                    trades = run_backtest(df, PAIR, LOT_SIZE, SL_PIPS, TP_PIPS, USE_SPREAD, USE_COMMISSION)
                    metrics = calculate_metrics(trades)
                    
                    if not trades:
                        st.warning("❌ No trades executed. Check your strategy logic.")
                    else:
                        if ENABLE_TELEGRAM:
                            msg = f"<b>RJ Algo Backtest Complete</b>\n\nPair: {PAIR}\nTrades: {metrics['total_trades']}\nWin Rate: {metrics['win_rate']:.1f}%\nNet P&L: ${metrics['net_pnl']:.2f}"
                            send_telegram(msg)
                        
                        st.markdown("---")
                        st.subheader("📊 Performance Dashboard")
                        
                        m1, m2, m3, m4, m5, m6 = st.columns(6)
                        m1.metric("Total Trades", metrics['total_trades'])
                        m2.metric("Win Rate", f"{metrics['win_rate']:.1f}%", f"{metrics['total_wins']}W / {metrics['total_losses']}L")
                        m3.metric("Net P&L", f"${metrics['net_pnl']:.2f}")
                        m4.metric("Max Drawdown", f"{metrics['max_dd']:.1f}%")
                        m5.metric("Sharpe Ratio", f"{metrics['sharpe']:.2f}")
                        m6.metric("Sortino Ratio", f"{metrics['sortino']:.2f}")
                        
                        m7, m8, m9 = st.columns(3)
                        m7.metric("Profit Factor", f"{metrics['profit_factor']:.2f}")
                        m8.metric("Avg Win", f"${metrics['avg_win']:.2f}")
                        m9.metric("Avg Loss", f"${metrics['avg_loss']:.2f}")
                        
                        st.session_state['df'] = df
                        st.session_state['trades'] = trades
                        st.session_state['metrics'] = metrics
                        st.session_state['strategy_name'] = selected_template
                        
                        st.markdown("---")
                        t1, t2 = st.tabs(["📋 Trade Log", "💰 Lot Simulator"])
                        with t1:
                            st.dataframe(pd.DataFrame(trades), use_container_width=True)
                        with t2:
                            pnls = [t['Pnl'] for t in trades]
                            lot_data = []
                            for lt in [0.01, 0.05, 0.10, 0.20, 0.50, 1.0, 2.0]:
                                mf = lt / LOT_SIZE if LOT_SIZE > 0 else 0
                                lot_data.append({
                                    "Lot": lt,
                                    "Risk/Trade": f"${abs(min(pnls)) * mf:.2f}",
                                    "Reward/Trade": f"${max(pnls) * mf:.2f}",
                                    "Net P&L": f"${metrics['net_pnl'] * mf:.2f}",
                                    "ROI": f"{(metrics['net_pnl'] * mf / 10000) * 100:.1f}%"
                                })
                            st.dataframe(pd.DataFrame(lot_data), use_container_width=True)
                        
                        st.success("✅ Backtest complete! Now go to '📈 Live Chart' tab to see your strategy visualized!")
                        
            except Exception as e:
                st.error(f"❌ Error: {e}")
                import traceback
                st.code(traceback.format_exc())

# ==================== TAB 2: LIVE CHART ====================
with tab2:
    st.header("📈 Real-Time Strategy Visualization")
    st.info("Yahan tumhari strategy LIVE chart pe dikhegi — Candlesticks, Buy/Sell arrows, SL/TP lines, Indicators — sab kuch!")
    
    if 'df' not in st.session_state or 'trades' not in st.session_state:
        st.warning("⚠️ Pehle '📊 Strategy & Backtest' tab mein 'RUN PRO BACKTEST' karo!")
    else:
        df = st.session_state['df']
        trades = st.session_state['trades']
        strategy_name = st.session_state.get('strategy_name', 'Custom')
        
        col1, col2 = st.columns([1, 3])
        with col1:
            st.subheader("Chart Settings")
            show_indicators = st.checkbox("Show Indicators (EMA, BB, etc.)", value=True)
            show_signals = st.checkbox("Show All Signals", value=True)
            chart_candles = st.slider("Max Candles", 50, 500, 300)
            st.caption(f"Showing last {min(chart_candles, len(df))} candles")
        
        with col2:
            st.subheader(f"{PAIR} | {strategy_name}")
            
            with st.spinner("Generating professional chart..."):
                fig = plot_strategy_chart(df, trades, PAIR, strategy_name, show_indicators)
                st.pyplot(fig)
            
            buf = io.BytesIO()
            fig.savefig(buf, format='png', dpi=150, bbox_inches='tight', facecolor='#0e1117')
            st.download_button("📥 Download Chart", buf.getvalue(), f"{PAIR}_strategy_chart.png", "image/png")
        
        st.markdown("---")
        st.subheader("📖 Chart Legend")
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.markdown("🟢 **Green Arrow Up** = BUY Entry")
        c2.markdown("🔴 **Red Arrow Down** = SELL Entry")
        c3.markdown("--- **Green Dashed** = Take Profit")
        c4.markdown("--- **Red Dashed** = Stop Loss")
        c5.markdown("💰 **$ Label** = Trade P&L")
        
        st.info("💡 Tip: Agar chart crowded lag raha hai, toh 'Show All Signals' off karo. Sirf executed trades dikhengi.")

# ==================== TAB 3: MONTE CARLO ====================
with tab3:
    st.header("🎲 Monte Carlo Simulation")
    if 'trades' not in st.session_state or not st.session_state['trades']:
        st.warning("⚠️ Pehle Backtest karo!")
    else:
        n_sims = st.slider("Simulations", 100, 5000, 1000, 100)
        if st.button("🎲 RUN MONTE CARLO", use_container_width=True):
            with st.spinner("Running..."):
                mc = monte_carlo_simulation(st.session_state['trades'], n_sims)
            if mc:
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Best Case", f"${mc['best']:,.2f}")
                c2.metric("Worst Case", f"${mc['worst']:,.2f}")
                c3.metric("Average", f"${mc['avg']:,.2f}")
                c4.metric("Ruin Probability", f"{mc['ruin_prob']:.1f}%",
                         delta="SAFE" if mc['ruin_prob'] < 5 else "RISKY")
                
                pnls = [t['Pnl'] for t in st.session_state['trades']]
                final_eqs = []
                for _ in range(min(n_sims, 500)):
                    random.shuffle(pnls)
                    eq = 10000
                    for p in pnls:
                        eq += p
                        if eq <= 5000:
                            break
                    final_eqs.append(eq)
                
                fig, ax = plt.subplots(figsize=(12, 5))
                fig.patch.set_facecolor('#0e1117')
                ax.hist(final_eqs, bins=50, color='#00d4ff', alpha=0.7, edgecolor='white')
                ax.axvline(x=10000, color='red', linestyle='--', linewidth=2, label='Start')
                ax.axvline(x=mc['avg'], color='#00ff88', linestyle='--', linewidth=2, label=f'Avg: ${mc["avg"]:,.0f}')
                ax.set_title('Final Equity Distribution', color='white', fontsize=14, fontweight='bold')
                ax.set_facecolor('#1e1e2e')
                ax.tick_params(colors='white')
                ax.legend(facecolor='#1e1e2e', edgecolor='white', labelcolor='white')
                ax.grid(True, alpha=0.2, color='gray')
                st.pyplot(fig)

# ==================== TAB 4: WALK FORWARD ====================
with tab4:
    st.header("🚶 Walk-Forward Analysis")
    if 'df' not in st.session_state:
        st.warning("⚠️ Pehle Backtest karo!")
    else:
        n_splits = st.slider("Splits", 2, 5, 3)
        if st.button("🚶 RUN WALK-FORWARD", use_container_width=True):
            with st.spinner("Analyzing..."):
                exec(strategy_code, globals())
                wf_df = walk_forward_analysis(st.session_state['df'], PAIR, LOT_SIZE,
                                               SL_PIPS, TP_PIPS, my_strategy, n_splits)
                st.dataframe(wf_df, use_container_width=True)
                st.success("✅ Consistent results across splits = Robust strategy (not curve-fitted)")

# ==================== TAB 5: OPTIMIZER ====================
with tab5:
    st.header("⚡ Auto SL/TP Optimizer")
    if 'df' not in st.session_state:
        st.warning("⚠️ Pehle Backtest karo!")
    else:
        if st.button("⚡ RUN OPTIMIZER", use_container_width=True):
            with st.spinner("Testing combinations..."):
                exec(strategy_code, globals())
                best_params, best_pnl, opt_df = optimize_strategy(
                    st.session_state['df'], PAIR, LOT_SIZE, my_strategy
                )
                if best_params:
                    st.success(f"🏆 Best: SL={best_params[0]} | TP={best_params[1]} | P&L=${best_pnl:.2f}")
                    pivot = opt_df.pivot(index='SL', columns='TP', values='PnL')
                    fig, ax = plt.subplots(figsize=(12, 8))
                    fig.patch.set_facecolor('#0e1117')
                    im = ax.imshow(pivot.values, cmap='RdYlGn', aspect='auto')
                    ax.set_xticks(range(len(pivot.columns)))
                    ax.set_xticklabels(pivot.columns, color='white')
                    ax.set_yticks(range(len(pivot.index)))
                    ax.set_yticklabels(pivot.index, color='white')
                    ax.set_xlabel('TP (Pips)', color='white')
                    ax.set_ylabel('SL (Pips)', color='white')
                    ax.set_title('P&L Heatmap', color='white', fontsize=14, fontweight='bold')
                    ax.set_facecolor('#1e1e2e')
                    plt.colorbar(im, ax=ax, label='P&L ($)')
                    st.pyplot(fig)
                    st.dataframe(opt_df.sort_values('PnL', ascending=False).head(10), use_container_width=True)

# ==================== TAB 6: ML PREDICTOR ====================
with tab6:
    st.header("🧠 AI Signal Predictor")
    if not SKLEARN_AVAILABLE:
        st.error("❌ scikit-learn not installed!")
        st.info("💡 Fix: Add 'scikit-learn' to your requirements.txt file, then restart the app.")
        st.code("""streamlit
pandas
numpy
yfinance
matplotlib
scikit-learn""", language='text')
    elif 'df' not in st.session_state:
        st.warning("⚠️ Pehle Backtest karo!")
    else:
        if st.button("🧠 TRAIN AI MODEL", use_container_width=True):
            with st.spinner("Training..."):
                result, error = ml_predictor(st.session_state['df'], PAIR)
                if error:
                    st.error(error)
                else:
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Accuracy", f"{result['accuracy']:.1f}%")
                    c2.metric("Prediction", result['prediction'])
                    c3.metric("Confidence", f"{result['confidence']:.1f}%")
                    st.dataframe(result['importance'], use_container_width=True)
                    fig, ax = plt.subplots(figsize=(10, 5))
                    fig.patch.set_facecolor('#0e1117')
                    ax.barh(result['importance']['Feature'], result['importance']['Importance'], color='#00d4ff')
                    ax.set_title('Feature Importance', color='white', fontsize=14, fontweight='bold')
                    ax.set_facecolor('#1e1e2e')
                    ax.tick_params(colors='white')
                    ax.grid(True, alpha=0.2, color='gray')
                    st.pyplot(fig)

# ==================== TAB 7: SETTINGS ====================
with tab7:
    st.header("📱 Telegram Configuration")
    st.info("Backtest complete hone pe phone pe instant alert")
    telegram_token = st.text_input("Bot Token", value=TELEGRAM_BOT_TOKEN, type="password")
    telegram_chat = st.text_input("Chat ID", value=TELEGRAM_CHAT_ID)
    if st.button("📨 Send Test"):
        if send_telegram("<b>Test Alert</b>\n\nRJ Algo Tools is working!"):
            st.success("✅ Sent! Check Telegram.")
        else:
            st.error("❌ Failed. Check token/ID.")
    st.markdown("---")
    st.header("🛠️ Spread Settings")
    st.json(SPREAD_MAP)
