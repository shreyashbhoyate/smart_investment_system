import pandas as pd

# =========================================================
# SECTOR FILTERING BASED ON USER PROFILE
# =========================================================
def recommend_sectors(sector_df, user):
    """
    Recommend sectors using ML-predicted returns and user constraints.

    Logic:
    - Filter sectors by risk appetite (Low / Medium / High)
    - Compute a score that prefers higher predicted returns and penalizes
      volatility for shorter horizons
    - Return top N sectors (default up to 6)
    """
    # Determine allowed risk classes
    if user.get("risk_appetite") == "Low":
        allowed = ["Low"]
    elif user.get("risk_appetite") == "Medium":
        allowed = ["Low", "Medium"]
    else:
        allowed = sector_df["risk_class"].unique().tolist()

    filtered = sector_df[sector_df["risk_class"].isin(allowed)].copy()
    if filtered.empty:
        return filtered

    # Use predicted_return and volatility to compute a ranking score
    # Shorter time horizons penalize volatility more.
    horizon = user.get("time_horizon", 3)
    if horizon <= 2:
        vol_penalty = 0.25
    elif horizon <= 5:
        vol_penalty = 0.12
    else:
        vol_penalty = 0.06

    # Ensure required columns exist
    if "predicted_return" not in filtered.columns:
        filtered["predicted_return"] = 0.0
    if "volatility" not in filtered.columns:
        filtered["volatility"] = filtered.get("volatility", 10.0)

    filtered["score"] = filtered["predicted_return"] - (filtered["volatility"] * vol_penalty)

    # Sort and select top sectors
    filtered = filtered.sort_values(by="score", ascending=False)
    top_n = min(6, len(filtered))
    return filtered.head(top_n)


# =========================================================
# PORTFOLIO ALLOCATION (FIXED + STABLE)
# =========================================================
def allocate_portfolio(filtered_sectors, user):
    """
    Creates portfolio allocation with:
    sector, risk, expected return, allocation %, amount (₹)
    """

    portfolio = []
    total_amount = user["investment_amount"]

    if len(filtered_sectors) == 0:
        return portfolio
    # Use ML-predicted returns to weight allocations rather than equal weight.
    # Higher predicted returns -> higher allocation. Apply a risk penalty
    # for conservative users so allocations align with risk appetite.

    scores = []
    rows = list(filtered_sectors.iterrows())
    for _, row in rows:
        pred_return = float(row.get("predicted_return", 0))
        # Ensure positive baseline so very low/negative returns don't get large weight
        base_score = max(pred_return, 0.01)

        # Risk penalty: penalize by sector risk_score when user is conservative
        risk_score = float(row.get("risk_score", 5))
        if user.get("risk_appetite") == "Low":
            penalty = 0.5 * risk_score
        elif user.get("risk_appetite") == "Medium":
            penalty = 0.2 * risk_score
        else:
            penalty = 0.0

        final_score = base_score - penalty
        # avoid non-positive scores
        if final_score <= 0:
            final_score = 0.01

        scores.append(final_score)

    # Normalize scores to percentages
    total_score = sum(scores)
    weights = [s / total_score for s in scores]

    # Build portfolio entries, ensure percentages sum to 100
    percents = [round(w * 100, 1) for w in weights]
    # Fix rounding error by adjusting the largest weight
    diff = 100.0 - sum(percents)
    if abs(diff) >= 0.1:
        # find index of max weight to adjust
        idx_max = percents.index(max(percents))
        percents[idx_max] = round(percents[idx_max] + diff, 1)

    for (i, (_, row)) in enumerate(rows):
        allocation_percent = percents[i]
        amount = round((allocation_percent / 100) * total_amount)
        portfolio.append({
            "sector": row["sector"],
            "risk": row["risk_class"],
            "expected_return": round(row.get("predicted_return", 0), 2),
            "allocation_percent": allocation_percent,
            "amount": amount
        })

    return portfolio


# =========================================================
# PORTFOLIO HEALTH SCORE (0–100)
# =========================================================
def calculate_portfolio_health(portfolio, user):
    """
    Calculates a health score based on:
    diversification, risk alignment, return quality, time horizon
    """

    score = 50  # base

    # Diversification
    if len(portfolio) >= 4:
        score += 20
    elif len(portfolio) == 3:
        score += 10

    # Risk alignment
    if user["risk_appetite"] == "High":
        score += 15
    elif user["risk_appetite"] == "Medium":
        score += 10
    else:
        score += 5

    # Return quality
    avg_return = sum(p["expected_return"] for p in portfolio) / len(portfolio)
    if avg_return >= 12:
        score += 15
    elif avg_return >= 9:
        score += 10

    # Time horizon
    if user["time_horizon"] >= 3:
        score += 10
    else:
        score += 5

    return min(score, 100)


# =========================================================
# REBALANCING RECOMMENDATIONS
# =========================================================
def generate_rebalancing_advice(portfolio, user):
    """
    Generates human-readable rebalancing suggestions.
    """

    advice = []

    high_risk_sectors = [p for p in portfolio if p["risk"] == "High"]

    if user["risk_appetite"] != "High" and high_risk_sectors:
        advice.append(
            "Reduce exposure to high-risk sectors and reallocate to stable assets."
        )

    if len(portfolio) < 4:
        advice.append(
            "Portfolio is concentrated. Adding more sectors may improve diversification."
        )

    if user["time_horizon"] <= 2:
        advice.append(
            "Short investment horizon detected. Consider higher allocation to low-volatility sectors."
        )

    if not advice:
        advice.append(
            "Portfolio is well balanced. Periodic monitoring is recommended."
        )

    return advice


# =========================================================
# SCENARIO SIMULATION — MARKET DROP
# =========================================================
def simulate_market_drop(portfolio, drop_percent=10):
    """
    Simulates downside impact if market drops by X%.
    """

    simulation = []

    for p in portfolio:
        stressed_return = round(p["expected_return"] - (drop_percent * 0.5), 2)

        simulation.append({
            "sector": p["sector"],
            "original_return": p["expected_return"],
            "stressed_return": stressed_return
        })

    return simulation
