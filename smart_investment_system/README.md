📊 Smart Portfolio Assistant

An AI-powered investment portfolio assistant that helps users build, analyze, and optimize investment portfolios based on risk appetite, time horizon, and market intelligence.

Built as a full-stack ML + Data Visualization platform using Flask, Machine Learning, and Plotly.

🚀 Key Features
🧑 Investor Profiling

User-driven input (age, income, investment amount, risk appetite, time horizon)

Dynamic investor profile generation

Personalized portfolio recommendations

📈 Portfolio Allocation Engine

Risk-aware sector allocation

Investment amount distribution

Diversification logic

Rebalancing suggestions

🤖 Machine Learning Intelligence

ML-based expected return prediction

Feature importance (model explainability)

Sector clustering (KMeans)

Correlation analysis (risk vs return factors)

📊 Interactive Visualizations

Portfolio allocation (donut chart)

Expected returns by sector

Risk vs return landscape

Feature importance bar chart

Correlation heatmap

🧪 Scenario Simulation

Market stress testing (e.g. “What if market drops 10%?”)

Impact analysis on portfolio returns

🎯 Portfolio Health Score

Dynamic health score (0–100)

Risk alignment feedback

Actionable investment insights

🛠 Tech Stack

Frontend

HTML5, CSS3

Plotly (interactive charts)

Responsive dashboard UI

Backend

Python

Flask

Jinja2 templates

Machine Learning

Scikit-learn

Random Forest (return prediction)

KMeans (sector clustering)

Data

Pandas, NumPy

CSV-based baseline dataset (extensible to live data)

🧱 Project Structure
smart_investment_system/
│
├── app.py
├── logic/
│   └── portfolio_rules.py
├── ml/
│   ├── return_model.py
│   └── risk_model.py
├── data/
│   ├── sector_data.csv
│   └── user_profile.csv
├── templates/
│   ├── index.html
│   └── dashboard.html
├── static/
│   └── css/
│       └── style.css
└── README.md

▶️ How to Run the Project
1️⃣ Install Dependencies
pip install flask pandas numpy scikit-learn plotly

2️⃣ Run the App
python app.py

3️⃣ Open in Browser
http://127.0.0.1:5000

📌 Use Cases

Personal investment planning

Portfolio risk analysis

Financial advisory dashboards

Hackathons & ML showcases

FinTech product prototyping

🔮 Future Enhancements

Live stock market data integration

User authentication

Historical backtesting

ETF-level recommendations

Cloud deployment (AWS / Azure)

Mobile-friendly UI