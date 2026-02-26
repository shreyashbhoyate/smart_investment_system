"""
Live Data Integration via Yahoo Finance
Fetches real-time stock data for portfolio sectors
Automatically converts all prices to INR (Indian Rupees)
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# ================= CURRENCY CONVERSION =================
USD_TO_INR = 83.0  # 1 USD = 83 INR (approximate market rate)

def is_nse_ticker(ticker):
    """Check if ticker is an Indian NSE stock (ends with .NS)"""
    return isinstance(ticker, str) and ticker.endswith('.NS')

def convert_price_to_inr(price, ticker):
    """
    Convert price to INR if needed.
    NSE stocks (.NS) are already in INR, so no conversion needed.
    US stocks need USD to INR conversion.
    """
    if is_nse_ticker(ticker):
        return float(price)  # Already in INR
    else:
        return float(price) * USD_TO_INR  # Convert USD to INR

# ================= SECTOR TO TICKER MAPPING =================
SECTOR_TICKERS = {
    "Technology": ["TCS.NS", "INFY.NS", "WIPRO.NS"],  # Tata Consultancy Services, Infosys, Wipro
    "Finance": ["HDFCBANK.NS", "ICICIBANK.NS", "AXISBANK.NS"],  # HDFC Bank, ICICI Bank, Axis Bank
    "Pharma": ["SUNPHARMA.NS", "CIPLA.NS", "DRREDDY.NS"],  # Sun Pharmaceutical, Cipla, Dr. Reddy's Laboratories
    "FMCG": ["ITC.NS", "NESTLEIND.NS", "BRITANNIA.NS"],  # ITC, Nestlé India, Britannia Industries
    "Energy": ["NTPC.NS", "ONGC.NS", "RELIANCE.NS"],  # NTPC, ONGC, Reliance Industries
    "Metals": ["TATASTEEL.NS", "HINDALCO.NS", "JSWSTEEL.NS"],  # Tata Steel, Hindalco Industries, JSW Steel
    "Auto": ["MARUTI.NS", "EICHERMOT.NS", "HEROMOTOCO.NS"],  # Maruti Suzuki, Eicher Motors, Hero MotoCorp
    "Telecom": ["BHARTIARTL.NS", "INDIGO.NS", "LT.NS"],  # Bharti Airtel, IndiGo, Larsen & Toubro
    "Real Estate": ["DLF.NS", "GODREJPROP.NS", "LODHA.NS"],  # DLF, Godrej Properties, Lodha Group
    "Utilities": ["POWERGRID.NS", "NTPC.NS", "TATAPOWER.NS"],  # Power Grid Corporation, NTPC, Tata Power
}

def fetch_sector_data():
    """
    Fetch live sector data from Yahoo Finance.
    Returns a DataFrame with sector metrics: avg_return, volatility, risk_score.
    Optimized for Indian NSE stocks (symbols ending with .NS)
    """
    sector_data = []
    
    for sector, tickers in SECTOR_TICKERS.items():
        try:
            # Download 1 year of historical data for each ticker
            end_date = datetime.now()
            start_date = end_date - timedelta(days=365)
            
            # Verify all tickers end with .NS (Indian NSE format)
            valid_tickers = [t for t in tickers if t.endswith('.NS')]
            if not valid_tickers:
                raise ValueError(f"No valid NSE tickers found for {sector}")
            
            print(f"📥 Fetching {sector} data: {valid_tickers}")
            
            # Download data with proper error handling
            data = yf.download(
                valid_tickers,
                start=start_date,
                end=end_date,
                progress=False
            )
            
            # Handle empty data
            if data.empty:
                print(f"⚠ No data retrieved for {sector}")
                raise ValueError("Empty data returned from Yahoo Finance")
            
            # Extract adjusted close prices properly
            # yfinance returns 'Close' not 'Adj Close' in latest versions
            try:
                if 'Close' in data.columns:
                    prices = data['Close']
                    if isinstance(prices, pd.Series):
                        prices = prices.to_frame()
                elif 'Adj Close' in data.columns:
                    prices = data['Adj Close']
                    if isinstance(prices, pd.Series):
                        prices = prices.to_frame()
                else:
                    # Try to extract from multi-index columns
                    price_cols = [col for col in data.columns if 'Close' in str(col)]
                    if price_cols:
                        prices = data[price_cols]
                    else:
                        raise ValueError(f"No price column found. Available: {data.columns.tolist()}")
            except Exception as col_err:
                print(f"⚠ Column extraction failed: {col_err}")
                raise
            
            # Calculate daily returns with proper handling
            daily_returns = prices.pct_change().dropna()
            
            if daily_returns.empty or daily_returns.isna().all().all():
                print(f"⚠ No valid returns for {sector}")
                raise ValueError("No valid return data")
            
            # Calculate sector metrics
            avg_return = float(daily_returns.mean().mean() * 252 * 100)  # Annualized return %
            volatility = float(daily_returns.std().mean() * np.sqrt(252) * 100)  # Annualized volatility %
            
            # Risk score: 1-10 based on volatility
            risk_score = float(min(10, max(1, (volatility / 5))))  # Normalize to 1-10
            
            sector_data.append({
                "sector": sector,
                "avg_return": round(avg_return, 2),
                "volatility": round(volatility, 2),
                "risk_score": round(risk_score, 2),
                "market_trend": "Growing" if avg_return > 0 else "Defensive",
                "market_trend_encoded": 2 if avg_return > 0 else 0,
                "latest_price": 0,
                "tickers": valid_tickers
            })
            
            print(f"✓ {sector}: Return {avg_return:.2f}% | Volatility {volatility:.2f}% | Risk {risk_score:.2f}/10")
            
        except Exception as e:
            print(f"✗ Error fetching {sector}: {e}")
            # Fallback to default values if API fails
            sector_data.append({
                "sector": sector,
                "avg_return": 8.5,
                "volatility": 15.0,
                "risk_score": 5.0,
                "market_trend": "Stable",
                "market_trend_encoded": 1,
                "latest_price": 0,
                "tickers": tickers
            })
    
    return pd.DataFrame(sector_data)


def fetch_historical_data(ticker, days=365):
    """
    Fetch historical data for a specific ticker.
    """
    try:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        data = yf.download(ticker, start=start_date, end=end_date, progress=False)
        return data
    except Exception as e:
        print(f"Error fetching data for {ticker}: {e}")
        return None


def get_sector_performance():
    """
    Get sector performance summary for dashboard display.
    """
    df = fetch_sector_data()
    return df.to_dict('records')


def fetch_sector_detailed_data(sector):
    """
    Fetch detailed data for a specific sector including historical prices.
    Returns sector metrics and historical price data for Indian NSE stocks.
    """
    if sector not in SECTOR_TICKERS:
        print(f"✗ Sector '{sector}' not found")
        return None
    
    tickers = SECTOR_TICKERS[sector]
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)
    
    print(f"🔍 Fetching detailed data for {sector}: {tickers}")
    
    try:
        # Download 1 year of data
        data = yf.download(
            tickers,
            start=start_date,
            end=end_date,
            progress=False
        )
        
        if data.empty:
            print(f"✗ No data for {sector}")
            return None
        
        # Extract Close prices with proper handling
        if 'Close' in data.columns:
            prices = data['Close']
            if isinstance(prices, pd.Series):
                prices = prices.to_frame()
        elif 'Adj Close' in data.columns:
            prices = data['Adj Close']
            if isinstance(prices, pd.Series):
                prices = prices.to_frame()
        else:
            # Try multi-index columns
            price_cols = [col for col in data.columns if 'Close' in str(col)]
            if price_cols:
                prices = data[price_cols]
            else:
                print(f"✗ No Close price column for {sector}")
                return None
        
        # Calculate metrics for each ticker
        ticker_metrics = {}
        for ticker in tickers:
            try:
                # Handle both Series and DataFrame column access
                if isinstance(prices, pd.DataFrame):
                    if ticker in prices.columns:
                        ticker_prices = prices[ticker]
                    else:
                        # Try to find column with ticker in it
                        matching_cols = [col for col in prices.columns if ticker in str(col)]
                        if not matching_cols:
                            print(f"  ⚠ {ticker}: No data column found")
                            continue
                        ticker_prices = prices[matching_cols[0]]
                else:
                    if ticker in prices.columns:
                        ticker_prices = prices[ticker]
                    else:
                        print(f"  ⚠ {ticker}: Not in price columns")
                        continue
                
                if ticker_prices is None or ticker_prices.empty:
                    print(f"  ⚠ {ticker}: Empty price data")
                    continue
                    
                returns = ticker_prices.pct_change().dropna()
                
                if not returns.empty:
                    # Convert prices to INR if needed (NSE stocks are already in INR)
                    current_price_inr = convert_price_to_inr(ticker_prices.iloc[-1], ticker)
                    previous_price_inr = convert_price_to_inr(ticker_prices.iloc[-2] if len(ticker_prices) > 1 else ticker_prices.iloc[-1], ticker)
                    high_52w_inr = convert_price_to_inr(ticker_prices.max(), ticker)
                    low_52w_inr = convert_price_to_inr(ticker_prices.min(), ticker)
                    
                    ticker_metrics[ticker] = {
                        "current_price": current_price_inr,
                        "previous_price": previous_price_inr,
                        "1y_return": float((ticker_prices.iloc[-1] / ticker_prices.iloc[0] - 1) * 100) if len(ticker_prices) > 0 else 0,
                        "volatility": float(returns.std() * np.sqrt(252) * 100),
                        "high_52w": high_52w_inr,
                        "low_52w": low_52w_inr,
                    }
                    print(f"  ✓ {ticker}: Price ₹{ticker_metrics[ticker]['current_price']:.2f} | Return {ticker_metrics[ticker]['1y_return']:.2f}%")
            except Exception as e:
                print(f"  ✗ Error processing {ticker}: {e}")
                continue
        
        # Aggregate sector data
        if not ticker_metrics:
            print(f"✗ No metrics calculated for {sector}")
            return None
        
        returns_list = [m["1y_return"] for m in ticker_metrics.values()]
        volatility_list = [m["volatility"] for m in ticker_metrics.values()]
        
        sector_data = {
            "sector": sector,
            "tickers": tickers,  # Keep as list, not string
            "ticker_metrics": ticker_metrics,
            "sector_avg_return": round(np.mean(returns_list), 2),
            "sector_volatility": round(np.mean(volatility_list), 2),
            "sector_risk_score": round(min(10, max(1, (np.mean(volatility_list) / 5))), 2),
            "price_history": prices.to_dict(),
            "dates": [str(d.date()) for d in prices.index]
        }
        
        print(f"✓ {sector} detailed data loaded successfully")
        return sector_data
        
    except Exception as e:
        print(f"✗ Error fetching detailed data for {sector}: {e}")
        return None
