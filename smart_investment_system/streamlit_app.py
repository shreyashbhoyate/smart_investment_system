import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# ML imports
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

# Import your existing modules
from logic.portfolio_rules import (
    recommend_sectors,
    allocate_portfolio,
    calculate_portfolio_health,
    generate_rebalancing_advice,
    simulate_market_drop
)
from live_data import fetch_sector_data

# PAGE CONFIG
st.set_page_config(
    page_title="Smart Portfolio Assistant",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CUSTOM CSS
st.markdown("""
<style>
    .main {
        background-color: #f0f4f8;
    }
    .stMetric {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    }
    h1 {
        color: #1f2937;
        font-weight: 700;
    }
</style>
""", unsafe_allow_html=True)

# LOAD & TRAIN ML MODEL
@st.cache_resource
def load_and_train_model():
    """Load data and train ML model - cached for performance"""
    
    try:
        sector_df = fetch_sector_data()
    except Exception as e:
        st.warning(f"Using cached data. Yahoo Finance error: {str(e)}")
        sector_df = pd.read_csv("data/sector_data.csv")
    
    # Encode market trend
    trend_map = {"Defensive": 0, "Stable": 1, "Growing": 2}
    sector_df["market_trend_encoded"] = sector_df["market_trend"].map(trend_map)
    
    # Risk classification
    def risk_label(score):
        if score <= 4:
            return "Low"
        elif score <= 6:
            return "Medium"
        else:
            return "High"
    
    sector_df["risk_class"] = sector_df["risk_score"].apply(risk_label)
    
    # Train ML models
    X = sector_df[["volatility", "risk_score", "market_trend_encoded"]]
    y = sector_df["avg_return"]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    
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
    
    return sector_df, best_model_name

# Load data
sector_df, model_name = load_and_train_model()

# SIDEBAR - USER INPUT
with st.sidebar:
    st.title("Smart Portfolio")
    st.markdown("---")
    
    st.subheader("Your Profile")
    
    name = st.text_input("Full Name", value="Investor")
    age = st.number_input("Age", min_value=18, max_value=100, value=30)
    investment = st.number_input(
        "Investment Amount (Rs)", 
        min_value=10000, 
        max_value=100000000, 
        value=160000,
        step=10000
    )
    
    risk = st.selectbox("Risk Appetite", ["Low", "Medium", "High"])
    horizon = st.slider("Time Horizon (Years)", min_value=1, max_value=30, value=5)
    
    st.markdown("---")
    analyze_button = st.button("Analyze Portfolio", type="primary", use_container_width=True)

# MAIN CONTENT
if analyze_button:
    st.session_state['analyzed'] = True
    st.session_state['user_data'] = {
        'name': name,
        'age': age,
        'investment': investment,
        'risk': risk,
        'horizon': horizon
    }

if st.session_state.get('analyzed', False):
    user_data = st.session_state['user_data']
    
    st.title(f"{user_data['name']}'s Investment Strategy")
    st.caption(f"AI-optimized portfolio for {user_data['risk']} risk investors")
    
    # Build portfolio
    user = {
        "investment_amount": user_data['investment'],
        "risk_appetite": user_data['risk'],
        "time_horizon": user_data['horizon']
    }
    
    filtered = recommend_sectors(sector_df, user)
    portfolio = allocate_portfolio(filtered, user)
    health_score = calculate_portfolio_health(portfolio, user)
    rebalancing_tips = generate_rebalancing_advice(portfolio, user)
    scenario_results = simulate_market_drop(portfolio, 10)
    
    total_return = sum(p["expected_return"] * p["allocation_percent"] / 100 for p in portfolio)
    projected_value = user_data['investment'] * (1 + total_return/100)
    
    # Metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Investment", f"Rs {user_data['investment']:,}", f"{user_data['horizon']} years")
    with col2:
        st.metric("Expected Return", f"{total_return:.2f}%", "Annually")
    with col3:
        st.metric("Projected Value", f"Rs {projected_value:,.0f}", f"+Rs {projected_value - user_data['investment']:,.0f}")
    with col4:
        st.metric("Health Score", f"{health_score}/100", "Excellent" if health_score >= 80 else "Good")
    
    st.markdown("---")
    
    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs(["Allocation", "Analysis", "Insights", "Details"])
    
    with tab1:
        st.subheader("Where to Invest Your Money")
        
        for p in portfolio:
            risk_color = "#10b981" if p["risk"] == "Low" else ("#f59e0b" if p["risk"] == "Medium" else "#ef4444")
            
            with st.container():
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.markdown(f"### {p['sector']}")
                    st.markdown(f"<span style='background:{risk_color}20;color:{risk_color};padding:4px 12px;border-radius:20px;font-size:0.85em;font-weight:600'>{p['risk']} Risk</span>", unsafe_allow_html=True)
                with col2:
                    st.markdown(f"<h2 style='text-align:right;color:#1e40af'>Rs {p['amount']:,}</h2>", unsafe_allow_html=True)
                
                c1, c2, c3 = st.columns(3)
                with c1:
                    st.metric("Portfolio Share", f"{p['allocation_percent']:.1f}%")
                with c2:
                    st.metric("Expected Return", f"{p['expected_return']:.2f}%")
                with c3:
                    st.metric("Potential Profit", f"+Rs {p['amount'] * p['expected_return'] / 100:,.0f}")
                
                st.markdown("---")
    
    with tab2:
        st.subheader("Portfolio Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig_pie = px.pie(
                values=[p['allocation_percent'] for p in portfolio],
                names=[p['sector'] for p in portfolio],
                title="Portfolio Allocation",
                hole=0.4
            )
            st.plotly_chart(fig_pie, use_container_width=True)
        
        with col2:
            fig_scatter = px.scatter(
                sector_df,
                x="volatility",
                y="predicted_return",
                size="risk_score",
                color="risk_class",
                hover_name="sector",
                title="Risk vs Return"
            )
            st.plotly_chart(fig_scatter, use_container_width=True)
        
        fig_bar = px.bar(
            sector_df.sort_values("predicted_return", ascending=False),
            x="sector",
            y="predicted_return",
            color="risk_class",
            title="Expected Returns by Sector"
        )
        st.plotly_chart(fig_bar, use_container_width=True)
    
    with tab3:
        st.subheader("Investment Insights")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.info(f"Diversification: Portfolio spread across {len(portfolio)} sectors")
            best = max(portfolio, key=lambda x: x['expected_return'])
            st.success(f"Best Performer: {best['sector']} - {best['expected_return']:.2f}%")
        
        with col2:
            st.warning(f"Risk Match: Suits {user_data['risk']} risk appetite")
            st.info(f"Time Horizon: Optimized for {user_data['horizon']} years")
        
        st.subheader("Recommendations")
        for tip in rebalancing_tips:
            st.markdown(f"- {tip}")
    
    with tab4:
        st.subheader("Detailed Breakdown")
        
        df_display = pd.DataFrame(portfolio)
        st.dataframe(df_display, use_container_width=True, hide_index=True)
        
        st.subheader("Stress Test: Market Drop 10%")
        df_stress = pd.DataFrame(scenario_results)
        st.dataframe(df_stress, use_container_width=True, hide_index=True)

else:
    st.title("Smart Portfolio Assistant")
    st.subheader("AI-Powered Investment Intelligence")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown(f"""
        ### Features:
        
        - Machine Learning: {model_name}
        - Risk-Optimized: Personalized allocation
        - Live Data: Yahoo Finance integration
        - Custom Strategy: Based on your goals
        
        ### Quick Start:
        
        1. Fill your profile in sidebar
        2. Click Analyze Portfolio
        3. Get instant recommendations
        
        **Sectors:** {len(sector_df)} | **Last Updated:** {datetime.now().strftime("%H:%M")}
        """)
    
    with col2:
        st.info("Start by filling your profile in the sidebar")

st.markdown("---")
st.caption("Built with ML - Python - Streamlit - Yahoo Finance")