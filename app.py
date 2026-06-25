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
@st.cache_data(ttl=3600)
def load_data(pair, timeframe):
    # Ticker mapping
    ticker_map = {
        'EURUSD': 'EURUSD=X',
        'GBPUSD': 'GBPUSD=X',
        'USDJPY': 'USDJPY=X',
        'XAUUSD': 'GC=F',      # Gold Futures
        'BTCUSD': 'BTC-USD'    # Bitcoin
    }
    
    # Period mapping based on timeframe
    period_map = {
        '1m': '7d',     # 1m data only available for 7 days
        '5m': '60d',    # 5m data for 60 days
        '15m': '60d',
        '1h': '730d',   # 1h data for 2 years
        '4h': '730d',
        '1d': '5y'      # Daily data for 5 years
    }
    
    ticker = ticker_map.get(pair, pair)
    period = period_map.get(timeframe, '1y')
    
    try:
        # Download data
        df = yf.download(
            ticker, 
            period=period, 
            interval=timeframe, 
            progress=False, 
            auto_adjust=False,
            threads=False
        )
        
        # Check if data is empty
        if df is None or df.empty:
            return None, "Data empty - try different timeframe"
        
        # Handle MultiIndex columns (new yfinance format)
        if isinstance(df.columns, pd.MultiIndex):
            # Flatten MultiIndex: ('Close', 'BTC-USD') -> 'Close'
            df.columns = df.columns.get_level_values(0)
        
        # Standardize column names
        column_mapping = {}
        for col in df.columns:
            col_str = str(col).upper()
            if 'OPEN' in col_str and 'ADJ' not in col_str:
                column_mapping[col] = 'Open'
            elif 'HIGH' in col_str:
                column_mapping[col] = 'High'
            elif 'LOW' in col_str:
                column_mapping[col] = 'Low'
            elif 'CLOSE' in col_str and 'ADJ' not in col_str:
                column_mapping[col] = 'Close'
            elif 'VOLUME' in col_str:
                column_mapping[col] = 'Volume'
        
        if column_mapping:
            df = df.rename(columns=column_mapping)
        
        # Ensure required columns exist
        required_cols = ['Open', 'High', 'Low', 'Close']
        missing_cols = [c for c in required_cols if c not in df.columns]
        
        if missing_cols:
            return None, f"Missing columns: {missing_cols}. Available: {df.columns.tolist()}"
        
        # Select only required columns and clean
        df = df[required_cols].copy()
        df = df.dropna()
        
        # Ensure numeric values
        for col in required_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        df = df.dropna()
        
        if len(df) < 50:
            return None, f"Only {len(df)} rows available. Need at least 50."
        
        return df, None
        
    except Exception as e:
        return None, str(e)

# ==================== BACKTEST ====================
if run_clicked:
    with st.spinner(f'Downloading {PAIR} ({TIMEFRAME}) data...'):
        df, error_msg = load_data(PAIR, TIMEFRAME)
    
    if df is None:
        st.error(f"❌ Data load nahi hua: {error_msg}")
        st.info("💡 Try karo: 1) Timeframe change karo (1h recommended) 2) Pair change karo (BTCUSD ya XAUUSD try karo)")
    else:
        st.success(f"✅ Data loaded: {len(df)} rows from {df.index[0].strftime('%Y-%m-%d')} to {df.index[-1].strftime('%Y-%m-%d')}")
        
        # Show sample data
        with st.expander("📊 Sample Data (First 5 rows)"):
            st.dataframe(df.head())
        
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
        
        # FIXED PIP MULTIPLIER
        if PAIR in ['EURUSD', 'GBPUSD']:
            mult = 0.0001
        elif PAIR == 'USDJPY':
            mult = 0.01
        else:
            mult = 0.10  # XAUUSD, BTCUSD
        
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
            st.warning("❌ Koi trade nahi hua is period me. Strategy ke rules ko check karo ya settings change karo.")
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
