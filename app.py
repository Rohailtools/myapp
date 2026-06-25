import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt

# --- PAGE CONFIG ---
st.set_page_config(page_title="Algo Backtester Pro", page_icon="📊", layout="wide")

# --- SYSTEM ENGINE ---
PAIRS = {'BTCUSD': 'BTC-USD', 'XAUUSD': 'GC=F', 'EURUSD': 'EURUSD=X', 'GBPUSD': 'GBPUSD=X', 'USDJPY': 'USDJPY=X'}
safe_limits = {'1m': '5d', '5m': '60d', '15m': '60d', '1h': '730d', '1d': 'max'}

def load_data(p, tf):
    try:
        df = yf.download(PAIRS.get(p), period=safe_limits.get(tf, '60d'), interval=tf, progress=False)
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.droplevel(1)
        df = df[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()
        return df if not df.empty else None
    except: return None

# --- DEFAULT STRATEGY ---
default_strategy = """def my_strategy(df):
    df['Signal'] = 0
    df['Trend'] = df['Close'].rolling(50).mean()
    df['Liq_High'] = df['High'].rolling(20).max().shift(1)
    df['Liq_Low'] = df['Low'].rolling(20).min().shift(1)
    
    bull_close = df['Close'] > df['Open']
    df.loc[(df['Close'] > df['Trend']) & (df['Low'] < df['Liq_Low']) & bull_close, 'Signal'] = 1
    bear_close = df['Open'] > df['Close']
    df.loc[(df['Close'] < df['Trend']) & (df['High'] > df['Liq_High']) & bear_close, 'Signal'] = -1
    return df"""

# --- FRONT UI DESIGN ---
st.title("📊 Algo Backtester Pro")
st.markdown("Made for Traders, by Traders.")

with st.sidebar:
    st.header("🛠️ Control Panel")
    PAIR = st.selectbox("Select Pair", ['XAUUSD', 'BTCUSD', 'EURUSD', 'GBPUSD', 'USDJPY'])
    TIMEFRAME = st.selectbox("Timeframe", ['1m', '5m', '15m', '1h', '1d'])
    LOT_SIZE = st.number_input("Lot Size", min_value=0.01, max_value=10.0, value=0.10, step=0.01)
    LEVERAGE = st.number_input("Leverage", min_value=1, max_value=2000, value=100)
    SL_PIPS = st.number_input("Stop Loss (Pips)", value=300)
    TP_PIPS = st.number_input("Take Profit (Pips)", value=600)
    
    st.markdown("---")
    st.header("✍️ Insert New Strategy")
    strategy_code = st.text_area("Paste YouTube Strategy Code Here:", height=200, value=default_strategy)
    
    run_clicked = st.button("🚀 RUN BACKTEST", use_container_width=True, type="primary")

if run_clicked:
    with st.spinner(f'Downloading {PAIR} ({TIMEFRAME}) data...'):
        df = load_data(PAIR, TIMEFRAME)
    
    if df is None:
        st.error("❌ Data load nahi hua. Timeframe ya Pair change karke try karo.")
    else:
        try:
            exec(strategy_code, globals())
            df = my_strategy(df)
        except Exception as e:
            st.error(f"❌ Strategy Code Error: {e}")
            st.stop()
            
        trades = []; in_trade = False; entry_p = sl_p = tp_p = 0; trade_type = 0
        pip_val = 0.1 if PAIR in ['EURUSD', 'GBPUSD', 'USDJPY'] else 1.0 

        for i in range(1, len(df)):
            if not in_trade:
                if df['Signal'].iloc[i] != 0:
                    in_trade=True; trade_type=df['Signal'].iloc[i]; entry_p=df['Open'].iloc[i]
                    mult = 0.0001 if PAIR in ['EURUSD','GBPUSD'] else (0.01 if PAIR in ['USDJPY','XAUUSD'] else 0.01)
                    sl_p=entry_p - (SL_PIPS*mult*trade_type)
                    tp_p=entry_p + (TP_PIPS*mult*trade_type)
            else:
                exit_p=0; reason=""
                if trade_type==1:
                    if df['Low'].iloc[i]<=sl_p: exit_p=sl_p; reason="SL Hit"
                    elif df['High'].iloc[i]>=tp_p: exit_p=tp_p; reason="TP Hit"
                    elif df['Signal'].iloc[i]==-1: exit_p=df['Open'].iloc[i]; reason="Signal Exit"
                else:
                    if df['High'].iloc[i]>=sl_p: exit_p=sl_p; reason="SL Hit"
                    elif df['Low'].iloc[i]<=tp_p: exit_p=tp_p; reason="TP Hit"
                    elif df['Signal'].iloc[i]==1: exit_p=df['Open'].iloc[i]; reason="Signal Exit"
                
                if exit_p!=0:
                    in_trade=False
                    mult = 0.0001 if PAIR in ['EURUSD','GBPUSD'] else (0.01 if PAIR in ['USDJPY','XAUUSD'] else 0.01)
                    pnl_pts = ((exit_p - entry_p) * trade_type) / mult
                    pnl_dol = pnl_pts * pip_val * LOT_SIZE
                    trades.append({'Type':'BUY' if trade_type==1 else 'SELL','Entry':round(entry_p,2),'Exit':round(exit_p,2),'Pnl':round(pnl_dol,2),'Reason':reason,'RRR':f"1:{TP_PIPS/SL_PIPS}" if reason=="TP Hit" else "1:0"})

        if not trades: 
            st.warning("❌ Koi trade nahi hua is period me.")
        else:
            wins = [t for t in trades if t['Pnl'] > 0]
            total_pnl = sum(t['Pnl'] for t in trades)
            wr = (len(wins) / len(trades)) * 100
            equity = [10000]
            for t in trades: equity.append(equity[-1] + t['Pnl'])
            max_dd = min([equity[i]-max(equity[:i+1]) for i in range(len(equity))])
            
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Total Trades", len(trades))
            col2.metric("Win Rate", f"{wr:.1f}%")
            col3.metric("Net P&L", f"${total_pnl:.2f}")
            col4.metric("Max Drawdown", f"${max_dd:.2f}")
            
            st.subheader("📈 Performance Graphs")
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6))
            ax1.plot(equity, color='#2a5298', linewidth=2)
            ax1.set_title('Equity Curve')
            dd_series = [(equity[i]-max(equity[:i+1])) for i in range(len(equity))]
            ax2.fill_between(range(len(dd_series)), dd_series, 0, color='red', alpha=0.4)
            ax2.set_title('Drawdown')
            plt.tight_layout()
            st.pyplot(fig)
            
            tab1, tab2 = st.tabs(["💰 Lot Adjuster", "📊 Trade Log"])
            
            with tab1:
                lot_data = []
                for lt in [0.01, 0.05, 0.10, 0.50, 1.0]:
                    mf = lt / LOT_SIZE
                    lot_data.append({"Lot": lt, "Risk $": round(abs(min(t['Pnl'] for t in trades)) * mf, 2), "Reward $": round(max(t['Pnl'] for t in trades) * mf, 2), "Est P&L $": round(total_pnl * mf, 2)})
                st.dataframe(pd.DataFrame(lot_data), use_container_width=True)
                
            with tab2:
                st.dataframe(pd.DataFrame(trades), use_container_width=True)
