from flask import Flask, render_template, request, redirect, url_for, session
import pandas as pd
import os

# ================= ML =================
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.cluster import KMeans

# ================= Visualization =================
import plotly.express as px
import plotly.io as pio
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
from datetime import timedelta, datetime
import functools
import time

# ================= CACHING =================
_cache = {}
_cache_time = {}

# ================= CURRENCY CONVERSION =================
USD_TO_INR = 83.0  # 1 USD = 83 INR (approximate market rate)

def convert_usd_to_inr(price_usd):
    """Convert USD price to INR"""
    return float(price_usd) * USD_TO_INR

# ================= Business Logic =================
from logic.portfolio_rules import (
    recommend_sectors,
    allocate_portfolio,
    calculate_portfolio_health,
    generate_rebalancing_advice,
    simulate_market_drop
)

# ================= Live Data Integration =================
from live_data import fetch_sector_data, fetch_sector_detailed_data

# ================= App Setup =================
app = Flask(__name__)
app.secret_key = 'smart_investment_portfolio_secret_2026'  # For session management
pio.templates.default = "plotly_dark"

# ================= Load Data =================
print("📡 Fetching live sector data from Yahoo Finance...")
try:
    sector_df = fetch_sector_data()
    print("✓ Live data loaded successfully!")
except Exception as e:
    print(f"⚠ Yahoo Finance error: {e}. Falling back to CSV data...")
    sector_df = pd.read_csv("data/sector_data.csv")

# ================= Encode Market Trend =================
trend_map = {"Defensive": 0, "Stable": 1, "Growing": 2}
sector_df["market_trend_encoded"] = sector_df["market_trend"].map(trend_map)

# ================= Risk Label =================
def risk_label(score):
    if score <= 4:
        return "Low"
    elif score <= 6:
        return "Medium"
    else:
        return "High"

sector_df["risk_class"] = sector_df["risk_score"].apply(risk_label)

# ================= ML Training =================
X = sector_df[["volatility", "risk_score", "market_trend_encoded"]]
y = sector_df["avg_return"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

lr = LinearRegression()
lr.fit(X_train, y_train)
lr_mae = mean_absolute_error(y_test, lr.predict(X_test))

rf = RandomForestRegressor(n_estimators=200, random_state=42)
rf.fit(X_train, y_train)
rf_mae = mean_absolute_error(y_test, rf.predict(X_test))

best_model = rf if rf_mae < lr_mae else lr
best_model_name = "Random Forest" if rf_mae < lr_mae else "Linear Regression"

best_model.fit(X, y)
sector_df["predicted_return"] = best_model.predict(X)

print(f"Best Model Selected: {best_model_name}")

# ================= FEATURE IMPORTANCE =================
importance = (
    best_model.feature_importances_
    if best_model_name == "Random Forest"
    else abs(best_model.coef_)
)

os.makedirs("static/images", exist_ok=True)

fi_df = pd.DataFrame({
    "Feature": X.columns,
    "Importance": importance
}).sort_values(by="Importance", ascending=False)

plt.figure(figsize=(8, 4))
sns.barplot(data=fi_df, x="Importance", y="Feature", hue="Feature", palette="viridis", legend=False)
plt.title("ML Feature Importance — Return Prediction")
plt.tight_layout()
plt.savefig("static/images/feature_importance.png")
plt.close()

# ================= KMEANS CLUSTERING =================
kmeans = KMeans(n_clusters=3, random_state=42)
sector_df["cluster"] = kmeans.fit_predict(
    sector_df[["predicted_return", "volatility"]]
)

cluster_map = {0: "Stable", 1: "Balanced Growth", 2: "High Growth"}
sector_df["cluster_label"] = sector_df["cluster"].map(cluster_map)

plt.figure(figsize=(7, 5))
sns.scatterplot(
    data=sector_df,
    x="volatility",
    y="predicted_return",
    hue="cluster_label",
    palette="Set2",
    s=120
)
plt.title("Sector Clustering — Risk vs Return")
plt.tight_layout()
plt.savefig("static/images/sector_clusters.png")
plt.close()

# ================= CORRELATION HEATMAP =================
corr = sector_df[
    ["avg_return", "predicted_return", "volatility", "risk_score", "market_trend_encoded"]
].corr()

plt.figure(figsize=(6, 5))
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Feature Correlation Matrix")
plt.tight_layout()
plt.savefig("static/images/correlation_heatmap.png")
plt.close()

# ================= CACHE HELPER =================
def cache_with_ttl(ttl_seconds=600):
    """Cache decorator with TTL (time-to-live in seconds)"""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            cache_key = f"{func.__name__}_{str(args)}_{str(kwargs)}"
            now = time.time()
            
            if cache_key in _cache and cache_key in _cache_time:
                if now - _cache_time[cache_key] < ttl_seconds:
                    return _cache[cache_key]
            
            result = func(*args, **kwargs)
            _cache[cache_key] = result
            _cache_time[cache_key] = now
            return result
        return wrapper
    return decorator


# ================= HELPER FUNCTIONS =================

@cache_with_ttl(ttl_seconds=600)
def get_live_sector_data():
    """
    Fetch live sector data with caching
    """
    global sector_df
    try:
        sector_df = fetch_sector_data()
        sector_df["market_trend_encoded"] = sector_df["market_trend"].map(trend_map)
        sector_df["risk_class"] = sector_df["risk_score"].apply(risk_label)
        
        # Re-train model with fresh data
        X_fresh = sector_df[["volatility", "risk_score", "market_trend_encoded"]]
        y_fresh = sector_df["avg_return"]
        best_model.fit(X_fresh, y_fresh)
        sector_df["predicted_return"] = best_model.predict(X_fresh)
        
        return sector_df
    except Exception as e:
        print(f"⚠ Error fetching live data: {e}")
        return sector_df


def build_portfolio(sector_data, investment_amount, risk_appetite):
    """
    Build portfolio based on user risk appetite
    """
    user = {
        "investment_amount": investment_amount,
        "risk_appetite": risk_appetite
    }
    
    # Recommend sectors based on risk
    filtered = recommend_sectors(sector_data, user)
    
    # Allocate portfolio
    portfolio = allocate_portfolio(filtered, user)
    
    return portfolio


def generate_allocation_pie(portfolio):
    """
    Generate portfolio allocation pie chart
    """
    portfolio_df = pd.DataFrame(portfolio)
    
    fig_pie = px.pie(
        portfolio_df,
        names="sector",
        values="allocation_percent",
        hole=0.45,
        title="Portfolio Allocation"
    )
    fig_pie.update_layout(title_x=0.5, height=400)
    return pio.to_html(fig_pie, full_html=False)


def generate_return_bar(sector_data):
    """
    Generate expected returns bar chart
    """
    fig_return = px.bar(
        sector_data.sort_values("predicted_return", ascending=False),
        x="sector",
        y="predicted_return",
        color="risk_class",
        text="predicted_return",
        title="Expected Annual Returns (%)"
    )
    fig_return.update_traces(texttemplate="%{text:.2f}%", textposition="outside")
    fig_return.update_layout(title_x=0.5, height=400)
    return pio.to_html(fig_return, full_html=False)


def generate_risk_return_plot(sector_data):
    """
    Generate risk vs return scatter plot
    """
    fig_risk = px.scatter(
        sector_data,
        x="volatility",
        y="predicted_return",
        color="risk_class",
        size="risk_score",
        hover_name="sector",
        title="Risk vs Return Landscape"
    )
    fig_risk.update_layout(title_x=0.5, height=400)
    return pio.to_html(fig_risk, full_html=False)


def generate_rebalancing_tips(portfolio, risk_appetite, time_horizon=5):
    """
    Generate rebalancing advice
    """
    user = {
        "risk_appetite": risk_appetite,
        "time_horizon": time_horizon
    }
    return generate_rebalancing_advice(portfolio, user)


def generate_investment_insights(portfolio, investment_amount, time_horizon):
    """
    Generate detailed, actionable investment insights showing:
    - How much to invest in each sector
    - Expected returns in ₹
    - Why each sector was selected
    - Total expected returns
    """
    insights = []
    total_expected_annual_return = 0
    
    for p in portfolio:
        sector = p['sector']
        amount = p['amount']
        allocation_pct = p['allocation_percent']
        expected_return_pct = p['expected_return']
        risk = p['risk']
        
        # Calculate expected annual return in ₹
        annual_return_amount = (amount * expected_return_pct) / 100
        total_expected_annual_return += annual_return_amount
        
        # Calculate returns over investment horizon
        total_return_amount = annual_return_amount * time_horizon
        final_value = amount + total_return_amount
        
        # Generate reasoning for why this sector was chosen
        if expected_return_pct >= 12:
            performance = "high growth potential"
        elif expected_return_pct >= 8:
            performance = "moderate growth"
        else:
            performance = "stable returns"
        
        reason = f"{risk} risk with {performance}"
        
        insights.append({
            'sector': sector,
            'invest_amount': f"₹{amount:,.0f}",
            'allocation_pct': f"{allocation_pct}%",
            'expected_annual_return_pct': f"{expected_return_pct}%",
            'expected_annual_return_amount': f"₹{annual_return_amount:,.0f}",
            'expected_total_return': f"₹{total_return_amount:,.0f}",
            'final_value': f"₹{final_value:,.0f}",
            'reason': reason,
            'risk_level': risk
        })
    
    # Calculate totals
    total_investment = sum(p['amount'] for p in portfolio)
    total_final_value = total_investment + (total_expected_annual_return * time_horizon)
    total_gain = total_expected_annual_return * time_horizon
    total_return_pct = (total_gain / total_investment) * 100
    
    summary = {
        'total_investment': f"₹{total_investment:,.0f}",
        'expected_annual_return': f"₹{total_expected_annual_return:,.0f}",
        'expected_total_gain': f"₹{total_gain:,.0f}",
        'expected_final_value': f"₹{total_final_value:,.0f}",
        'total_return_pct': f"{total_return_pct:.1f}%",
        'time_horizon': time_horizon
    }
    
    return {
        'insights': insights,
        'summary': summary
    }


# ================= ROUTES =================

# -------- ENTRY PAGE (USER INPUT) --------
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        try:
            # Debug: print all form data
            print("=" * 50)
            print("📥 FORM SUBMISSION RECEIVED")
            print("=" * 50)
            print(f"Form keys: {list(request.form.keys())}")
            print(f"Form values: {dict(request.form)}")
            print("=" * 50)
            
            # Get each field safely
            name = request.form.get("name", "").strip()
            age_str = request.form.get("age", "")
            investment_str = request.form.get("investment_amount", "")
            risk = request.form.get("risk_appetite", "")
            horizon_str = request.form.get("time_horizon", "")
            
            print(f"name: '{name}'")
            print(f"age_str: '{age_str}'")
            print(f"investment_str: '{investment_str}'")
            print(f"risk: '{risk}'")
            print(f"horizon_str: '{horizon_str}'")
            
            # Validate name
            if not name:
                print("❌ Name is empty")
                return render_template("index.html", error="Please enter your name")
            
            # Validate age
            if not age_str:
                print("❌ Age is empty")
                return render_template("index.html", error="Please enter your age")
            try:
                age = int(age_str)
                print(f"✓ Age converted: {age}")
            except ValueError:
                print(f"❌ Age is not a number: {age_str}")
                return render_template("index.html", error="Age must be a number")
            
            # Validate investment
            if not investment_str:
                print("❌ Investment is empty")
                return render_template("index.html", error="Please enter investment amount")
            try:
                investment = float(investment_str)
                print(f"✓ Investment converted: {investment}")
            except ValueError:
                print(f"❌ Investment is not a number: {investment_str}")
                return render_template("index.html", error="Investment amount must be a number")
            
            # Validate risk
            if not risk or risk == "":
                print("❌ Risk is empty")
                return render_template("index.html", error="Please select your risk appetite")
            print(f"✓ Risk selected: {risk}")
            
            # Validate horizon
            if not horizon_str:
                print("❌ Horizon is empty")
                return render_template("index.html", error="Please enter time horizon")
            try:
                horizon = int(horizon_str)
                print(f"✓ Horizon converted: {horizon}")
            except ValueError:
                print(f"❌ Horizon is not a number: {horizon_str}")
                return render_template("index.html", error="Time horizon must be a number")
            
            print("=" * 50)
            print(f"✅ ALL VALIDATIONS PASSED")
            print(f"✓ Name: {name}")
            print(f"✓ Age: {age}")
            print(f"✓ Investment: {investment}")
            print(f"✓ Risk: {risk}")
            print(f"✓ Horizon: {horizon}")
            print("=" * 50)
            
            # Store in session
            session['user_params'] = {
                'name': name,
                'age': age,
                'investment': investment,
                'risk': risk,
                'horizon': horizon
            }
            print("✓ Stored in session")
            
            # Redirect to dashboard
            redirect_url = url_for('dashboard', 
                                   name=name,
                                   age=age, 
                                   investment=investment, 
                                   risk=risk, 
                                   horizon=horizon)
            
            print(f"🔄 Redirecting to: {redirect_url}")
            print("=" * 50)
            return redirect(redirect_url)
            
        except Exception as e:
            print("=" * 50)
            print(f"❌❌❌ UNEXPECTED ERROR: {str(e)}")
            print(f"Error type: {type(e)}")
            import traceback
            traceback.print_exc()
            print("=" * 50)
            return render_template("index.html", error=f"Error: {str(e)}")

    # GET request - just show the form
    return render_template("index.html")


# -------- DASHBOARD (PERSISTENT VIEW) --------
@app.route("/dashboard", methods=["GET", "POST"])
def dashboard():
    """
    Display dashboard - accepts both form POST and navigation GET
    """
    try:
        # Check if coming from form POST or URL GET
        if request.method == "POST":
            # Form submission
            name = request.form.get("name", "Investor").strip()
            age = int(request.form.get("age"))
            investment = float(request.form.get("investment_amount"))
            risk = request.form.get("risk_appetite")
            horizon = int(request.form.get("time_horizon"))
            
            print(f"✓ Form POST to dashboard: name={name}, age={age}, risk={risk}, investment={investment}")
        else:
            # Navigation via URL parameters
            name = request.args.get('name', 'Investor')
            age = request.args.get('age', type=int)
            investment = request.args.get('investment', type=float)
            risk = request.args.get('risk', default='Medium')
            horizon = request.args.get('horizon', type=int)
            
            print(f"✓ GET to dashboard: name={name}, age={age}, risk={risk}, investment={investment}")
        
        if not all([age, investment, horizon]):
            return render_template('index.html', error='⚠️ Please fill the form first to create your portfolio.')
        
        # Store user params in session for "My Portfolio" feature
        session['user_params'] = {
            'name': name,
            'age': age,
            'investment': investment,
            'risk': risk,
            'horizon': horizon
        }
        
        # Rebuild portfolio with same parameters
        print(f"📊 Building dashboard for {name}: Investment=₹{investment}, Risk={risk}")
        
        live_sector_df = get_live_sector_data()
        portfolio = build_portfolio(live_sector_df, investment, risk)
        
        # Create complete user dictionary with all keys that might be needed
        user_data = {
            "age": age,
            "investment_amount": investment,
            "risk_appetite": risk,
            "time_horizon": horizon
        }
        
        print(f"🔍 DEBUG: user_data keys: {user_data.keys()}")
        print(f"🔍 DEBUG: user_data values: {user_data}")
        
        # Try different function signatures for calculate_portfolio_health
        try:
            print("🔍 Attempting: calculate_portfolio_health(portfolio, user_data)")
            health_score = calculate_portfolio_health(portfolio, user_data)
            print(f"✓ Health score calculated: {health_score}")
        except Exception as e:
            print(f"❌ First attempt failed: {str(e)}")
            try:
                print("🔍 Attempting: calculate_portfolio_health(portfolio=portfolio, user=user_data)")
                health_score = calculate_portfolio_health(portfolio=portfolio, user=user_data)
                print(f"✓ Health score calculated: {health_score}")
            except Exception as e2:
                print(f"❌ Second attempt failed: {str(e2)}")
                try:
                    print("🔍 Attempting: calculate_portfolio_health(portfolio, risk, horizon)")
                    health_score = calculate_portfolio_health(
                        portfolio=portfolio,
                        risk_level=risk,
                        horizon=horizon
                    )
                    print(f"✓ Health score calculated: {health_score}")
                except Exception as e3:
                    print(f"❌ Third attempt failed: {str(e3)}")
                    # Fallback: return a default score
                    print("⚠️ Using default health score of 75")
                    health_score = 75
        
        rebalancing_tips = generate_rebalancing_tips(portfolio, risk, horizon)
        scenario_results = simulate_market_drop(portfolio, 10)
        
        # Generate detailed investment insights
        investment_insights = generate_investment_insights(portfolio, investment, horizon)
        
        graph_pie = generate_allocation_pie(portfolio)
        graph_return = generate_return_bar(live_sector_df)
        graph_risk = generate_risk_return_plot(live_sector_df)
        
        user = {
            "name": name,
            "age": age,
            "investment": investment,
            "investment_amount": investment,
            "risk": risk,
            "risk_appetite": risk,
            "horizon": horizon,
            "time_horizon": horizon
        }
        
        print(f"✓ Dashboard generated successfully with {len(portfolio)} sectors")
        
        return render_template(
            "dashboard.html",
            user=user,
            portfolio=portfolio,
            health_score=health_score,
            rebalancing_tips=rebalancing_tips,
            scenario_results=scenario_results,
            investment_insights=investment_insights,
            graph_pie=graph_pie,
            graph_return=graph_return,
            graph_risk=graph_risk
        )
    except Exception as e:
        print(f"❌ Dashboard Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return render_template('index.html', error=f'Error building dashboard: {str(e)}')


# -------- USER PROFILE --------
@app.route("/profile", methods=["GET", "POST"])
def profile():
    """
    Display user profile information
    """
    # Get parameters from URL
    age = request.args.get('age', type=int)
    income = request.args.get('income', type=float)
    investment = request.args.get('investment', type=float)
    risk = request.args.get('risk', default='moderate')
    horizon = request.args.get('horizon', type=int)
    
    if not all([age, income, investment, horizon]):
        return redirect(url_for('index'))
    
    user = {
        "age": age,
        "income": income,
        "investment": investment,
        "risk": risk,
        "horizon": horizon
    }
    
    # If you have a profile.html template, use it
    # Otherwise, redirect to dashboard
    try:
        return render_template("profile.html", user=user)
    except:
        return redirect(url_for('dashboard', age=age, income=income, 
                               investment=investment, risk=risk, horizon=horizon))


# -------- MY PORTFOLIO (REDIRECT TO LAST DASHBOARD) --------
@app.route("/my-portfolio", methods=["GET"])
def my_portfolio():
    """
    Redirect to dashboard using saved session parameters
    """
    if 'user_params' not in session:
        return render_template('index.html', error='⚠️ Please create your portfolio first by filling the form!')
    
    params = session['user_params']
    return redirect(url_for('dashboard', 
                           name=params.get('name', 'Investor'),
                           age=params['age'],
                           investment=params['investment'],
                           risk=params['risk'],
                           horizon=params['horizon']))



# -------- SECTOR WATCHLIST --------
@app.route("/watchlist", methods=["GET", "POST"])
def watchlist():
    """
    Display all sectors with live data and performance metrics
    """
    live_sector_df = get_live_sector_data()
    
    # Check if user has created a portfolio
    has_portfolio = 'user_params' in session
    
    sector_list = []
    for _, row in live_sector_df.iterrows():
        sector_list.append({
            "sector": row["sector"],
            "avg_return": row["avg_return"],
            "volatility": row["volatility"],
            "risk_score": row["risk_score"],
            "risk_class": row.get("risk_class", "Medium"),
            "market_trend": row.get("market_trend", "Stable"),
        })
    
    # Create overview chart
    fig_overview = px.scatter(
        live_sector_df,
        x="volatility",
        y="avg_return",
        size="risk_score",
        color="risk_class",
        hover_name="sector",
        title="📊 Live Sector Overview — Risk vs Return",
        labels={"volatility": "Volatility (%)", "avg_return": "Annual Return (%)"}
    )
    fig_overview.update_layout(title_x=0.5, height=500)
    graph_overview = pio.to_html(fig_overview, full_html=False)
    
    # Create return comparison chart
    fig_returns = px.bar(
        live_sector_df.sort_values("avg_return", ascending=False),
        x="sector",
        y="avg_return",
        color="risk_class",
        text="avg_return",
        title="📈 Sector Returns Comparison"
    )
    fig_returns.update_traces(texttemplate="%{text:.2f}%", textposition="outside")
    fig_returns.update_layout(title_x=0.5, height=400)
    graph_returns = pio.to_html(fig_returns, full_html=False)
    
    return render_template(
        "watchlist.html",
        sectors=sector_list,
        graph_overview=graph_overview,
        graph_returns=graph_returns,
        has_portfolio=has_portfolio
    )


# -------- SECTOR DETAILS --------
@app.route("/sector/<sector_name>", methods=["GET", "POST"])
def sector_detail(sector_name):
    """
    Display detailed live data and charts for a specific sector
    """
    sector_data = fetch_sector_detailed_data(sector_name)
    
    if not sector_data:
        return f"Sector '{sector_name}' not found or no data available", 404
    
    # Create price history chart with normalized data
    fig_sector = px.line(
        title=f"📈 {sector_name} — 1-Year Performance (Normalized)"
    )
    
    # Get historical data for each ticker
    if sector_data.get("ticker_metrics"):
        for ticker in sector_data["tickers"]:
            try:
                # Download historical data
                hist_data = yf.download(
                    ticker,
                    start=datetime.now() - timedelta(days=365),
                    end=datetime.now(),
                    progress=False
                )
                
                if hist_data.empty:
                    print(f"⚠ No data for {ticker}")
                    continue
                
                # Extract Close prices
                if 'Close' in hist_data.columns:
                    prices = hist_data['Close']
                else:
                    continue
                
                # Normalize prices to percentage change from first day (starting at 100)
                normalized_prices = ((prices / prices.iloc[0]) * 100).values.tolist()
                dates = [str(d.date()) for d in prices.index]
                
                # Add trace to chart
                fig_sector.add_scatter(
                    x=dates,
                    y=normalized_prices,
                    name=f"{ticker}",
                    mode='lines',
                    hovertemplate='<b>%{fullData.name}</b><br>Date: %{x}<br>Normalized Index: %{y:.1f}<extra></extra>'
                )
                
            except Exception as e:
                print(f"✗ Error fetching {ticker}: {e}")
                continue
    
    fig_sector.update_layout(
        title_x=0.5,
        height=500,
        hovermode='x unified',
        xaxis_title="Date",
        yaxis_title="Normalized Price Index (Base = 100)",
        template="plotly_dark",
        plot_bgcolor='#1f2933',
        paper_bgcolor='#0f172a'
    )
    graph_sector = pio.to_html(fig_sector, full_html=False)
    
    return render_template(
        "sector_detail.html",
        sector_name=sector_name,
        sector_data=sector_data,
        graph_sector=graph_sector
    )


# ================= DEBUG ROUTES =================
@app.route('/_routes')
def _routes():
    """Debug: list registered routes"""
    routes = []
    for rule in app.url_map.iter_rules():
        routes.append(f"{rule.methods} {rule.rule}")
    return "<pre>" + "\n".join(sorted(routes)) + "</pre>"


# ================= RUN =================
if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)