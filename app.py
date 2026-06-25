import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt

st.set_page_config(page_title="RJ Algo Tools", layout="wide")
st.title("🚀 RJ Algo Tools - Backtester")

# ==================== SIDEBAR ====================
st.sidebar.header("⚙️ Settings")

PAIR = st.sidebar.selectbox("Pair", ['EURUSD', 'GBPUSD', 'USDJPY', 'XAUUSD', 'BTCUSD'])
TIMEFRAME = st.sidebar.selectbox("Timeframe", ['1m', '5m', '15m', '1h', '4h', '1d'])
LOT_SIZE = st.sidebar.number_input("Lot Size", min_value=0.01, value=0.10, step=0.01)
LEVERAGE = st.sidebar.number_input("Leverage", min_value=1, max_value=2000, value=100)
SL_PIPS = st.sidebar.number_input("Stop Loss (Pips)", value=300)
TP_PIPS = st.sidebar.number_input("Take Profit (Pips)", value=600)

# ==================== STRATEGY INPUT ====================
st.markdown("---")
st.header("✍️ Insert New Strategy")

default_strategy = '''def my_strategy(df):
    df['Signal'] = 0
    
    # --- EMA 50 ---
    df['EMA50'] = df['Close'].ewm(span=50, adjust=False).mean()
    
    # --- Liquidity High/Low (20 period) ---
    df['Liq_High'] = df['High'].rolling(window=20).max().shift(1)
    df['Liq_Low'] = df['Low'].rolling(window=20).min().shift(1)
    
    # --- Candle Types ---
    df['Bull'] = df['Close'] > df['Open']
    df['Bear'] = df['Open'] > df['Close']
    
    # --- BUY: Price > EMA50, Low sweeps Liq_Low, Bullish close ---
    df.loc[(df['Close'] > df['EMA50']) & (df['Low'] < df['Liq_Low']) & (df['Bull']), 'Signal'] = 1
    
    # --- SELL: Price < EMA50, High sweeps Liq_High, Bearish close ---
    df.loc[(df['Close'] < df['EMA50']) & (df['High'] > df['Liq_High']) & (df['Bear']), 'Signal'] = -1
    
    return df'''

strategy_code = st.text_area("Paste YouTube Strategy Code Here:", height=250, value=default_strategy)

run_clicked = st.button("🚀 RUN BACKTEST", use_container_width=True, type="primary")

# ==================== DATA LOADER ====================
@st.cache_data
def load_data(pair, timeframe):
    ticker_map = {
        'EURUSD': 'EURUSD=X',
        'GBPUSD': 'GBPUSD=X',
        'USDJPY': 'USDJPY=X',
        'XAUUSD': 'GC=F',
        'BTCUSD': 'BTC-USD'
    }
    ticker = ticker_map.get(pair, pair)
    
    tf_map = {'1m':'1m', '5m':'5m', '15m':'15m', '1h':'1h', '4h':'4h', '1d':'1d'}
    period = "60d" if timeframe in ['1m', '5m'] else "730d"
    
    try:
        df = yf.download(ticker, period=period, interval=tf_map[timeframe], progress=False, auto_adjust=False)
        if df.empty:
            return None
        
        # Flatten MultiIndex columns if present
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        
        # Rename columns to standard names
        col_map = {}
        for c in df.columns:
            if 'Open' in str(c):
                col_map[c] = 'Open'
            elif 'High' in str(c):
                col_map[c] = 'High'
            elif 'Low' in str(c):
                col_map[c] = 'Low'
            elif 'Close' in str(c) and 'Adj' not in str(c):
                col_map[c] = 'Close'
            elif 'Volume' in str(c):
                col_map[c] = 'Volume'
        
        if col_map:
            df = df.rename(columns=col_map)
        
        # Ensure required columns exist
        required = ['Open', 'High', 'Low', 'Close']
        for col in required:
            if col not in df.columns:
                return None
        
        df = df[required].copy()
        df.dropna(inplace=True)
        return df
    except Exception as e:
        return None

# ==================== BACKTEST ====================
if run_clicked:
    with st.spinner(f'Downloading {PAIR} ({TIMEFRAME}) data...'):
        df = load_data(PAIR, TIMEFRAME)
    
    if df is None or df.empty:
        st.error("❌ Data load nahi hua. Timeframe ya Pair change karke try karo.")
    else:
        # Execute user strategy
        try:
            exec(strategy_code, globals())
            df = my_strategy(df)
            if 'Signal' not in df.columns:
                st.error("❌ Strategy me 'Signal' column return karni zaroori hai! (1=Buy, -1=Sell, 0=No Trade)")
                st.stop()
        except Exception as e:
            st.error(f"❌ Strategy Code Error: {e}")
            st.stop()
        
        # Backtest engine
        trades = []
        in_trade = False
        entry_p = 0
        sl_p = 0
        tp_p = 0
        trade_type = 0
        
        # ✅ FIXED PIP MULTIPLIER (XAUUSD = 0.10)
        if PAIR in ['EURUSD', 'GBPUSD']:
            mult = 0.0001
        elif PAIR == 'USDJPY':
            mult = 0.01
        else:
            mult = 0.10  # XAUUSD, BTCUSD, etc.
        
        pip_val = 0.1 if PAIR in ['EURUSD', 'GBPUSD', 'USDJPY'] else 1.0
        
        for i in range(1, len(df)):
            if not in_trade:
                if df['Signal'].iloc[i] != 0:
                    in_trade = True
                    trade_type = int(df['Signal'].iloc[i])
                    entry_p = float(df['Open'].iloc[i])
                    sl_p = entry_p - (SL_PIPS * mult * trade_type)
                    tp_p = entry_p + (TP_PIPS * mult * trade_type)
            else:
                exit_p = 0
                reason = ""
                
                if trade_type == 1:  # BUY
                    if df['Low'].iloc[i] <= sl_p:
                        exit_p = sl_p
                        reason = "SL Hit"
                    elif df['High'].iloc[i] >= tp_p:
                        exit_p = tp_p
                        reason = "TP Hit"
                    elif df['Signal'].iloc[i] == -1:
                        exit_p = float(df['Open'].iloc[i])
                        reason = "Signal Exit"
                else:  # SELL
                    if df['High'].iloc[i] >= sl_p:
                        exit_p = sl_p
                        reason = "SL Hit"
                    elif df['Low'].iloc[i] <= tp_p:
                        exit_p = tp_p
                        reason = "TP Hit"
                    elif df['Signal'].iloc[i] == 1:
                        exit_p = float(df['Open'].iloc[i])
                        reason = "Signal Exit"
                
                if exit_p != 0:
                    in_trade = False
                    pnl_pts = ((exit_p - entry_p) * trade_type) / mult
                    pnl_dol = pnl_pts * pip_val * LOT_SIZE
                    trades.append({
                        'Type': 'BUY' if trade_type == 1 else 'SELL',
                        'Entry': round(entry_p, 2),
                        'Exit': round(exit_p, 2),
                        'Pnl': round(pnl_dol, 2),
                        'Reason': reason,
                        'RRR': f"1:{TP_PIPS/SL_PIPS:.1f}" if reason == "TP Hit" else "1:0"
                    })
        
        # ==================== RESULTS ====================
        if not trades:
            st.warning("❌ Koi trade nahi hua is period me.")
        else:
            wins = [t for t in trades if t['Pnl'] > 0]
            total_pnl = sum(t['Pnl'] for t in trades)
            wr = (len(wins) / len(trades)) * 100
            equity = [10000]
            for t in trades:
                equity.append(equity[-1] + t['Pnl'])
            max_dd = min([equity[i] - max(equity[:i+1]) for i in range(len(equity))])
            
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Total Trades", len(trades))
            col2.metric("Win Rate", f"{wr:.1f}%")
            col3.metric("Net P&L", f"${total_pnl:.2f}")
            col4.metric("Max Drawdown", f"${max_dd:.2f}")
            
            st.subheader("📈 Performance Graphs")
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6))
            ax1.plot(equity, color='#2a5298', linewidth=2)
            ax1.set_title('Equity Curve')
            ax1.grid(True, alpha=0.3)
            dd_series = [(equity[i] - max(equity[:i+1])) for i in range(len(equity))]
            ax2.fill_between(range(len(dd_series)), dd_series, 0, color='red', alpha=0.4)
            ax2.set_title('Drawdown')
            ax2.grid(True, alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig)
            
            tab1, tab2 = st.tabs(["💰 Lot Adjuster", "📊 Trade Log"])
            
            with tab1:
                lot_data = []
                for lt in [0.01, 0.05, 0.10, 0.50, 1.0]:
                    mf = lt / LOT_SIZE if LOT_SIZE > 0 else 0
                    lot_data.append({
                        "Lot": lt,
                        "Risk $": round(abs(min(t['Pnl'] for t in trades)) * mf, 2),
                        "Reward $": round(max(t['Pnl'] for t in trades) * mf, 2),
                        "Est P&L $": round(total_pnl * mf, 2)
                    })
                st.dataframe(pd.DataFrame(lot_data), use_container_width=True)
                
            with tab2:
                st.dataframe(pd.DataFrame(trades), use_container_width=True)
