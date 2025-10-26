import pandas as pd
import numpy as np
import sqlite3
from ta.trend import SMAIndicator, EMAIndicator, MACD
from ta.momentum import RSIIndicator, ROCIndicator, StochasticOscillator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.volume import OnBalanceVolumeIndicator

def load_data_from_db():
    """Load data from SQLite database"""
    conn = sqlite3.connect('nifty_gap.db')
    
    nifty = pd.read_sql('SELECT * FROM nifty_data', conn)
    vix = pd.read_sql('SELECT * FROM vix_data', conn)
    sp500 = pd.read_sql('SELECT * FROM sp500_data', conn)
    
    conn.close()
    
    print(f"\n📊 Raw Data Loaded:")
    print(f"   Nifty: {len(nifty)} records")
    print(f"   VIX: {len(vix)} records")
    print(f"   S&P500: {len(sp500)} records")
    
    return nifty, vix, sp500

def calculate_gap(df):
    """Calculate gap: Today's Open - Yesterday's Close"""
    df = df.copy()
    df['Date'] = pd.to_datetime(df['Date'])
    df['Previous_Close'] = df['Close'].shift(1)
    df['Gap'] = df['Open'] - df['Previous_Close']
    df['Gap_Percent'] = (df['Gap'] / df['Previous_Close']) * 100
    df['Gap_Direction'] = (df['Gap'] > 0).astype(int)
    return df

def add_vix_signal(nifty_df, vix_df):
    """Add VIX change signal"""
    nifty_df = nifty_df.copy()
    vix_df = vix_df.copy()
    
    nifty_df['Date'] = pd.to_datetime(nifty_df['Date'])
    vix_df['Date'] = pd.to_datetime(vix_df['Date'])
    
    vix_df['VIX_Change'] = vix_df['Close'].diff()
    vix_df['VIX_Direction'] = (vix_df['VIX_Change'] < 0).astype(int)
    
    # Keep only needed columns
    vix_df = vix_df[['Date', 'Close', 'VIX_Change', 'VIX_Direction']]
    vix_df = vix_df.rename(columns={'Close': 'Close_VIX'})
    
    merged = pd.merge(nifty_df, vix_df, on='Date', how='left')
    
    print(f"   After VIX merge: {len(merged)} records (dropped {len(nifty_df) - len(merged)})")
    
    return merged

def add_candlestick_signal(df):
    """Add candlestick signal"""
    df = df.copy()
    df['Candle_Color'] = (df['Close'] > df['Open']).astype(int)
    df['Candle_Size'] = abs(df['Close'] - df['Open'])
    df['Candle_Size_Percent'] = (df['Candle_Size'] / df['Open']) * 100
    df['Upper_Shadow'] = df['High'] - df['Close']
    df['Lower_Shadow'] = df['Open'] - df['Low']
    return df

def add_sp500_signal(nifty_df, sp500_df):
    """Add S&P 500 overnight momentum"""
    nifty_df = nifty_df.copy()
    sp500_df = sp500_df.copy()
    
    sp500_df['Date'] = pd.to_datetime(sp500_df['Date'])
    sp500_df['SP_Change'] = sp500_df['Close'].pct_change() * 100
    sp500_df['SP_Direction'] = (sp500_df['SP_Change'] > 0).astype(int)
    
    # Shift date by 1 day (S&P closes before Nifty opens next day)
    sp500_df['Date'] = sp500_df['Date'] + pd.Timedelta(days=1)
    
    sp500_df = sp500_df[['Date', 'SP_Change', 'SP_Direction']]
    
    merged = pd.merge(nifty_df, sp500_df, on='Date', how='left')
    
    print(f"   After S&P500 merge: {len(merged)} records (dropped {len(nifty_df) - len(merged)})")
    
    return merged

def add_basic_indicators(df):
    """Add basic technical indicators"""
    print("   Adding basic indicators (10)...")
    df = df.copy()
    
    try:
        rsi = RSIIndicator(close=df['Close'], window=14)
        df['RSI_14'] = rsi.rsi()
        
        sma20 = SMAIndicator(close=df['Close'], window=20)
        df['SMA_20'] = sma20.sma_indicator()
        
        sma50 = SMAIndicator(close=df['Close'], window=50)
        df['SMA_50'] = sma50.sma_indicator()
        
        ema12 = EMAIndicator(close=df['Close'], window=12)
        df['EMA_12'] = ema12.ema_indicator()
        
        macd = MACD(close=df['Close'])
        df['MACD'] = macd.macd()
        df['MACD_Signal'] = macd.macd_signal()
        df['MACD_Diff'] = macd.macd_diff()
        
        bb = BollingerBands(close=df['Close'], window=20, window_dev=2)
        df['BB_High'] = bb.bollinger_hband()
        df['BB_Low'] = bb.bollinger_lband()
        
        atr = AverageTrueRange(high=df['High'], low=df['Low'], close=df['Close'], window=14)
        df['ATR_14'] = atr.average_true_range()
    except Exception as e:
        print(f"      ⚠️  Error adding basic indicators: {e}")
    
    return df

def add_momentum_indicators(df):
    """Add momentum indicators"""
    print("   Adding momentum indicators (15)...")
    df = df.copy()
    
    try:
        roc5 = ROCIndicator(close=df['Close'], window=5)
        df['ROC_5'] = roc5.roc()
        
        roc10 = ROCIndicator(close=df['Close'], window=10)
        df['ROC_10'] = roc10.roc()
        
        roc20 = ROCIndicator(close=df['Close'], window=20)
        df['ROC_20'] = roc20.roc()
        
        stoch = StochasticOscillator(high=df['High'], low=df['Low'], close=df['Close'], window=14, smooth_window=3)
        df['Stoch_K'] = stoch.stoch()
        df['Stoch_D'] = stoch.stoch_signal()
        
        obv = OnBalanceVolumeIndicator(close=df['Close'], volume=df['Volume'])
        df['OBV'] = obv.on_balance_volume()
        
        df['Return_1d'] = df['Close'].pct_change() * 100
        df['Return_5d'] = df['Close'].pct_change(5) * 100
        df['Return_10d'] = df['Close'].pct_change(10) * 100
    except Exception as e:
        print(f"      ⚠️  Error adding momentum indicators: {e}")
    
    return df

def add_volatility_lagged_features(df):
    """Add volatility and lagged features"""
    print("   Adding volatility & lagged features (12)...")
    df = df.copy()
    
    try:
        if 'Return_1d' in df.columns:
            df['Volatility_5d'] = df['Return_1d'].rolling(5).std()
            df['Volatility_10d'] = df['Return_1d'].rolling(10).std()
            df['Volatility_20d'] = df['Return_1d'].rolling(20).std()
        
        if 'Close_VIX' in df.columns:
            df['VIX_Lag1'] = df['Close_VIX'].shift(1)
            df['VIX_Lag2'] = df['Close_VIX'].shift(2)
            df['VIX_MA_5'] = df['Close_VIX'].rolling(5).mean()
        
        if 'Gap_Percent' in df.columns:
            df['Gap_Lag1'] = df['Gap_Percent'].shift(1)
            df['Gap_Lag2'] = df['Gap_Percent'].shift(2)
            df['Gap_Lag3'] = df['Gap_Percent'].shift(3)
    except Exception as e:
        print(f"      ⚠️  Error adding volatility features: {e}")
    
    return df

def add_price_ratio_features(df):
    """Add price ratio features"""
    print("   Adding price ratio features (8)...")
    df = df.copy()
    
    try:
        if 'SMA_20' in df.columns:
            df['Price_to_SMA20'] = df['Close'] / df['SMA_20']
        if 'SMA_50' in df.columns:
            df['Price_to_SMA50'] = df['Close'] / df['SMA_50']
        
        df['HL_Range'] = (df['High'] - df['Low']) / df['Close']
        df['OC_Range'] = abs(df['Close'] - df['Open']) / df['Close']
        
        df['Close_to_High'] = (df['Close'] - df['Low']) / (df['High'] - df['Low'])
        
        df['Volume_MA_Ratio'] = df['Volume'] / df['Volume'].rolling(20).mean()
        df['Volume_Change'] = df['Volume'].pct_change() * 100
    except Exception as e:
        print(f"      ⚠️  Error adding price ratio features: {e}")
    
    return df

def add_rolling_stats(df):
    """Add rolling statistics"""
    print("   Adding rolling statistics (8)...")
    df = df.copy()
    
    try:
        if 'Return_1d' in df.columns:
            df['Return_MA_5'] = df['Return_1d'].rolling(5).mean()
            df['Return_MA_10'] = df['Return_1d'].rolling(10).mean()
            df['Return_Max_10'] = df['Return_1d'].rolling(10).max()
            df['Return_Min_10'] = df['Return_1d'].rolling(10).min()
        
        if 'Volatility_5d' in df.columns:
            df['Volatility_of_Volatility'] = df['Volatility_5d'].rolling(5).std()
        
        df['Range_20d'] = (df['High'].rolling(20).max() - df['Low'].rolling(20).min())
    except Exception as e:
        print(f"      ⚠️  Error adding rolling stats: {e}")
    
    return df

def add_candlestick_patterns(df):
    """Add candlestick pattern features"""
    print("   Adding candlestick patterns (8)...")
    df = df.copy()
    
    try:
        df['Doji'] = (abs(df['Close'] - df['Open']) / (df['High'] - df['Low'])) < 0.1
        df['Hammer'] = ((df['High'] - df['Low']) / df['Close']) > 0.3
        
        if 'Candle_Color' in df.columns and 'Candle_Size' in df.columns:
            df['Bull_Engulfing'] = (df['Candle_Color'] == 1) & (df['Candle_Color'].shift(1) == 0) & \
                                    (df['Candle_Size'] > df['Candle_Size'].shift(1))
            
            df['Bear_Engulfing'] = (df['Candle_Color'] == 0) & (df['Candle_Color'].shift(1) == 1) & \
                                    (df['Candle_Size'] > df['Candle_Size'].shift(1))
        
        if 'Gap_Percent' in df.columns:
            avg_gap = df['Gap_Percent'].rolling(20).mean()
            df['Gap_vs_Avg'] = df['Gap_Percent'] / (avg_gap + 0.001)
            
            df['Consecutive_Ups'] = (df['Return_1d'] > 0).rolling(5).sum() if 'Return_1d' in df.columns else 0
    except Exception as e:
        print(f"      ⚠️  Error adding candlestick patterns: {e}")
    
    return df

def add_vix_advanced_features(df):
    """Add advanced VIX features"""
    print("   Adding VIX advanced features (6)...")
    df = df.copy()
    
    try:
        if 'Close_VIX' in df.columns:
            df['VIX_MA_10'] = df['Close_VIX'].rolling(10).mean()
            df['VIX_MA_20'] = df['Close_VIX'].rolling(20).mean()
            
            df['VIX_Deviation'] = df['Close_VIX'] - df['VIX_MA_20']
            df['VIX_ROC'] = df['Close_VIX'].pct_change() * 100
            df['VIX_Spike'] = (df['Close_VIX'] > (df['VIX_MA_20'] * 1.2)).astype(int)
    except Exception as e:
        print(f"      ⚠️  Error adding VIX advanced features: {e}")
    
    return df

def add_sp500_advanced_features(df):
    """Add advanced S&P 500 features"""
    print("   Adding S&P500 advanced features (8)...")
    df = df.copy()
    
    try:
        if 'SP_Change' in df.columns:
            df['SP_Return_2d'] = df['SP_Change'].rolling(2).sum()
            df['SP_Return_5d'] = df['SP_Change'].rolling(5).sum()
            
            df['SP_MA_5'] = df['SP_Change'].rolling(5).mean()
            df['SP_MA_10'] = df['SP_Change'].rolling(10).mean()
            
            df['SP_Volatility'] = df['SP_Change'].rolling(10).std()
            
            if 'Return_1d' in df.columns:
                df['SP_Nifty_Correlation'] = df['SP_Change'].rolling(20).corr(df['Return_1d'])
            
            df['SP_Strength'] = (df['SP_Direction'] * df['SP_Change'].abs()) if 'SP_Direction' in df.columns else 0
    except Exception as e:
        print(f"      ⚠️  Error adding S&P500 advanced features: {e}")
    
    return df

def create_features():
    """Main function to create all features"""
    print("="*60)
    print("ADVANCED FEATURE ENGINEERING (v2 - Robust)")
    print("="*60)
    
    print("\n1️⃣  Loading data from database...")
    nifty, vix, sp500 = load_data_from_db()
    
    print("\n2️⃣  Calculating gaps...")
    nifty = calculate_gap(nifty)
    print(f"   After gap calc: {len(nifty)} records")
    
    print("\n3️⃣  Adding cross-market signals...")
    nifty = add_vix_signal(nifty, vix)
    nifty = add_candlestick_signal(nifty)
    nifty = add_sp500_signal(nifty, sp500)
    print(f"   After all merges: {len(nifty)} records")
    
    print("\n4️⃣  PHASE 1: Basic indicators...")
    nifty = add_basic_indicators(nifty)
    
    print("\n5️⃣  PHASE 2: Momentum indicators...")
    nifty = add_momentum_indicators(nifty)
    
    print("\n6️⃣  PHASE 3: Volatility & lagged features...")
    nifty = add_volatility_lagged_features(nifty)
    
    print("\n7️⃣  PHASE 4: Price ratio features...")
    nifty = add_price_ratio_features(nifty)
    
    print("\n8️⃣  PHASE 5: Rolling statistics...")
    nifty = add_rolling_stats(nifty)
    
    print("\n9️⃣  PHASE 6: Candlestick patterns...")
    nifty = add_candlestick_patterns(nifty)
    
    print("\n🔟 PHASE 7: VIX advanced features...")
    nifty = add_vix_advanced_features(nifty)
    
    print("\n1️⃣1️⃣ PHASE 8: S&P500 advanced features...")
    nifty = add_sp500_advanced_features(nifty)
    
    print("\n1️⃣2️⃣ Cleaning data...")
    before_clean = len(nifty)
    
    # Fill NaN values forward/backward
    nifty = nifty.fillna(method='ffill').fillna(method='bfill').fillna(0)
    
    # Replace inf with 0
    nifty = nifty.replace([np.inf, -np.inf], 0)
    
    # Remove rows where essential columns are all NaN
    nifty = nifty.dropna(subset=['Close', 'Gap_Direction', 'Candle_Color'], how='all')
    
    print(f"   Before clean: {before_clean}, After clean: {len(nifty)}")
    
    # Create target variable (next day's gap)
    nifty['Target'] = nifty['Gap_Direction'].shift(-1)
    nifty = nifty.dropna(subset=['Target'])
    
    print(f"   After target creation: {len(nifty)}")
    
    print("\n1️⃣3️⃣ Saving to database...")
    conn = sqlite3.connect('nifty_gap.db')
    nifty.to_sql('features_advanced', conn, if_exists='replace', index=False)
    conn.close()
    
    print(f"\n✅ Features created successfully!")
    print(f"   Total records: {len(nifty)}")
    print(f"   Total features: {len(nifty.columns)}")
    print(f"   Date range: {nifty['Date'].min()} to {nifty['Date'].max()}")
    
    return nifty

if __name__ == "__main__":
    df = create_features()
    print("\n" + "="*60)
    print("📊 Sample data:")
    print(df[['Date', 'Close', 'Gap_Direction', 'Target', 'RSI_14', 'MACD']].tail(5))