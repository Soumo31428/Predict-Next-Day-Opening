import yfinance as yf
import pandas as pd
import sqlite3

def collect_nifty_data():
    """Download Nifty50 historical data"""
    print("Downloading Nifty50 data...")
    nifty = yf.download("^NSEI", period="4y", interval="1d", progress=False)
    
    # Fix: Flatten multi-level columns if they exist
    if isinstance(nifty.columns, pd.MultiIndex):
        nifty.columns = nifty.columns.get_level_values(0)
    
    nifty = nifty.reset_index()
    
    # Ensure column names are clean
    nifty.columns = [col.strip() for col in nifty.columns]
    
    print(f"   ✅ Downloaded {len(nifty)} records")
    print(f"   Columns: {list(nifty.columns)}")
    
    return nifty

def collect_india_vix():
    """Download India VIX data"""
    print("Downloading India VIX data...")
    vix = yf.download("^INDIAVIX", period="4y", interval="1d", progress=False)
    
    # Fix: Flatten multi-level columns if they exist
    if isinstance(vix.columns, pd.MultiIndex):
        vix.columns = vix.columns.get_level_values(0)
    
    vix = vix.reset_index()
    vix.columns = [col.strip() for col in vix.columns]
    
    print(f"   ✅ Downloaded {len(vix)} records")
    
    return vix

def collect_sp500_data():
    """Download S&P 500 for overnight signal"""
    print("Downloading S&P 500 data...")
    sp500 = yf.download("^GSPC", period="4y", interval="1d", progress=False)
    
    # Fix: Flatten multi-level columns if they exist
    if isinstance(sp500.columns, pd.MultiIndex):
        sp500.columns = sp500.columns.get_level_values(0)
    
    sp500 = sp500.reset_index()
    sp500.columns = [col.strip() for col in sp500.columns]
    
    print(f"   ✅ Downloaded {len(sp500)} records")
    
    return sp500

def save_to_database(nifty, vix, sp500):
    """Save data to SQLite database"""
    print("\nSaving to database...")
    conn = sqlite3.connect('nifty_gap.db')
    
    nifty.to_sql('nifty_data', conn, if_exists='replace', index=False)
    vix.to_sql('vix_data', conn, if_exists='replace', index=False)
    sp500.to_sql('sp500_data', conn, if_exists='replace', index=False)
    
    conn.close()
    print("✅ Data saved to database successfully!")

if __name__ == "__main__":
    print("="*60)
    print("STEP 1: DATA COLLECTION")
    print("="*60)
    print()
    
    nifty = collect_nifty_data()
    vix = collect_india_vix()
    sp500 = collect_sp500_data()
    
    save_to_database(nifty, vix, sp500)
    
    print(f"\n{'='*60}")
    print("📊 SUMMARY")
    print(f"{'='*60}")
    print(f"   Nifty50 records: {len(nifty)}")
    print(f"   India VIX records: {len(vix)}")
    print(f"   S&P 500 records: {len(sp500)}")
    print(f"\n✅ Database created: nifty_gap.db")
    print(f"{'='*60}")
