"""
Simple Trading Simulation - A Basic Machine-Learning Application Using Streamlit

Beginner-friendly workflow:
  synthetic data -> train/test split -> preprocessing -> logistic regression
  -> evaluation -> interactive prediction -> simple simulated portfolio.

All data is FICTIONAL. Nothing here is financial advice.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

SEED = 42
N_RECORDS = 800
POSITIVE = "Profitable trade"
NEGATIVE = "Loss-making trade"

st.set_page_config(page_title="Trading Simulation", page_icon="📈", layout="wide")

NUMERIC_COLS = [
    "rsi",
    "ma_gap_pct",
    "volatility_pct",
    "volume_change_pct",
    "sentiment_score",
    "risk_per_trade_pct",
    "holding_days",
]
CATEGORICAL_COLS = ["asset", "market_condition", "experience_level"]


# ---------------------------------------------------------------------------
# 1. Synthetic data
# ---------------------------------------------------------------------------
@st.cache_data
def generate_data(n: int = N_RECORDS, seed: int = SEED) -> pd.DataFrame:
    """Create fictional trade records using a transparent rule plus noise."""
    rng = np.random.default_rng(seed)

    asset = rng.choice(["ALPHA", "BETA", "GAMMA", "DELTA", "OMEGA"], n)
    market_condition = rng.choice(["Bullish", "Sideways", "Bearish"], n, p=[0.40, 0.35, 0.25])
    experience_level = rng.choice(["Beginner", "Intermediate", "Advanced"], n, p=[0.40, 0.40, 0.20])

    rsi = np.clip(rng.normal(50, 15, n), 10, 90)
    ma_gap_pct = rng.normal(0, 3, n)  # short moving average vs long moving average (%)
    volatility_pct = np.clip(rng.gamma(2.5, 0.9, n), 0.5, 8)
    volume_change_pct = rng.normal(0, 25, n)
    sentiment_score = np.clip(np.round(rng.normal(5.5, 2.0, n)), 1, 10)
    risk_per_trade_pct = np.round(rng.uniform(0.5, 5.0, n), 1)
    holding_days = rng.integers(1, 31, n)

    market_bonus = pd.Series(market_condition).map(
        {"Bullish": 1.0, "Sideways": 0.0, "Bearish": -1.0}
    ).to_numpy()
    experience_bonus = pd.Series(experience_level).map(
        {"Beginner": -0.3, "Intermediate": 0.0, "Advanced": 0.4}
    ).to_numpy()

    # Fictional rule: trend, sentiment and market help; volatility, over-bought
    # RSI and oversized risk hurt. Noise keeps the problem from being perfect.
    score = (
        0.35 * ma_gap_pct
        + 0.30 * (sentiment_score - 5.5)
        + 0.80 * market_bonus
        + experience_bonus
        - 0.45 * (volatility_pct - 2.2)
        - 0.03 * np.maximum(rsi - 70, 0) * 3
        - 0.15 * (risk_per_trade_pct - 2.5)
        + 0.01 * volume_change_pct
    )
    score = score + rng.normal(0, 1.0, n)

    # Simulated percentage return of each trade (used by the portfolio simulation).
    return_pct = np.round(1.2 * score + rng.normal(0, 1.5, n), 2)
    target = np.where(return_pct > 0, POSITIVE, NEGATIVE)

    df = pd.DataFrame(
        {
            "asset": asset,
            "market_condition": market_condition,
            "experience_level": experience_level,
            "rsi": np.round(rsi, 1),
            "ma_gap_pct": np.round(ma_gap_pct, 2),
            "volatility_pct": np.round(volatility_pct, 2),
            "volume_change_pct": np.round(volume_change_pct, 1),
            "sentiment_score": sentiment_score.astype(int),
            "risk_per_trade_pct": risk_per_trade_pct,
            "holding_days": holding_days,
            "trade_return_pct": return_pct,
            "trade_outcome": target,
        }
    )
    return df


# ---------------------------------------------------------------------------
# 2. Model
# ---------------------------------------------------------------------------
@st.cache_resource
def train_model():
    df = generate_data()
    X = df[NUMERIC_COLS + CATEGORICAL_COLS]
    y = (df["trade_outcome"] == POSITIVE).astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=SEED, stratify=y
    )

    preprocess = ColumnTransformer(
        [
            (
                "num",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler()),
                    ]
                ),
                NUMERIC_COLS,
            ),
            (
                "cat",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                CATEGORICAL_COLS,
            ),
        ]
    )
    model = Pipeline(
        [("prep", preprocess), ("clf", LogisticRegression(max_iter=1000, random_state=SEED))]
    )
    model.fit(X_train, y_train)

    pred = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]
    metrics = {
        "Accuracy": accuracy_score(y_test, pred),
        "ROC-AUC": roc_auc_score(y_test, proba),
        "Precision": precision_score(y_test, pred),
        "Recall": recall_score(y_test, pred),
        "F1-score": f1_score(y_test, pred),
    }
    cm = confusion_matrix(y_test, pred)
    report = classification_report(
        y_test, pred, target_names=[NEGATIVE, POSITIVE], output_dict=False
    )
    return model, X_test, y_test, pred, proba, metrics, cm, report


# ---------------------------------------------------------------------------
# 3. Simple portfolio simulation
# ---------------------------------------------------------------------------
def simulate_portfolio(returns: pd.Series, start_capital: float, position_pct: float) -> np.ndarray:
    """Compound capital through a sequence of trades, risking position_pct each time."""
    equity = [start_capital]
    for r in returns:
        capital = equity[-1]
        equity.append(capital + capital * (position_pct / 100) * (r / 100))
    return np.array(equity)


# ---------------------------------------------------------------------------
# 4. Interface
# ---------------------------------------------------------------------------
df = generate_data()
model, X_test, y_test, pred, proba, metrics, cm, report = train_model()

st.title("📈 Simple Trading Simulation")
st.caption("A basic machine-learning application using Streamlit")
st.warning(
    "**Important:** This project uses synthetic data only. It contains no real market data "
    "and must not be used as financial advice or a real trading tool."
)

tab1, tab2, tab3, tab4 = st.tabs(
    ["Overview", "Explore data", "Prediction demo", "Model results"]
)

# ---- Tab 1: Overview -------------------------------------------------------
with tab1:
    st.header("Project objective")
    st.markdown(
        f"""
**Research question:** Can basic technical, sentiment, and risk variables predict a
synthetically generated *"{POSITIVE}"* label?

The app generates **{N_RECORDS} fictional trades**, trains a **logistic-regression**
model, shows exploratory charts, lets you test a fictional trade, and simulates how a
small portfolio would behave if it only took the trades the model approves.

**How it works**
1. Generate reproducible synthetic data (random seed {SEED}).
2. Split into 80% training and 20% testing data.
3. Impute missing values, standardize numbers, one-hot encode categories.
4. Train logistic regression and evaluate it on unseen test data.
5. Simulate a portfolio from the test trades.

**Responsible use:** the target follows a known fictional rule, so strong performance is
expected and says nothing about real markets. Real trading involves risk of loss.
"""
    )
    c1, c2, c3 = st.columns(3)
    c1.metric("Fictional trades", len(df))
    c2.metric("Features", len(NUMERIC_COLS) + len(CATEGORICAL_COLS))
    c3.metric("Profitable share", f"{(df['trade_outcome'] == POSITIVE).mean():.1%}")

# ---- Tab 2: Explore data ---------------------------------------------------
with tab2:
    st.header("Explore the fictional data")
    col = st.selectbox("Choose a variable to plot", NUMERIC_COLS + CATEGORICAL_COLS)
    fig, ax = plt.subplots(figsize=(8, 4))
    if col in NUMERIC_COLS:
        sns.histplot(data=df, x=col, hue="trade_outcome", kde=True, bins=25, ax=ax)
    else:
        sns.countplot(data=df, x=col, hue="trade_outcome", ax=ax)
    ax.set_title(f"{col} by trade outcome")
    st.pyplot(fig)
    plt.close(fig)

    st.subheader("Correlation of numeric variables")
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    sns.heatmap(
        df[NUMERIC_COLS + ["trade_return_pct"]].corr(), annot=True, fmt=".2f", cmap="coolwarm", ax=ax2
    )
    st.pyplot(fig2)
    plt.close(fig2)

    st.subheader("Fictional records")
    st.dataframe(df.head(50), width="stretch")

# ---- Tab 3: Prediction demo ------------------------------------------------
with tab3:
    st.header("Prediction demo")
    st.write("Enter one **fictional** trade setup.")
    a, b, c = st.columns(3)
    with a:
        asset = st.selectbox("Asset", ["ALPHA", "BETA", "GAMMA", "DELTA", "OMEGA"])
        market = st.selectbox("Market condition", ["Bullish", "Sideways", "Bearish"])
        exp = st.selectbox("Experience level", ["Beginner", "Intermediate", "Advanced"])
        rsi = st.slider("RSI", 10, 90, 50)
    with b:
        ma_gap = st.slider("Moving-average gap (%)", -10.0, 10.0, 0.0, 0.1)
        vol = st.slider("Volatility (%)", 0.5, 8.0, 2.0, 0.1)
        vchg = st.slider("Volume change (%)", -80, 80, 0)
    with c:
        sent = st.slider("Sentiment score (1-10)", 1, 10, 5)
        risk = st.slider("Risk per trade (%)", 0.5, 5.0, 2.0, 0.1)
        hold = st.slider("Holding days", 1, 30, 10)

    if st.button("Predict outcome"):
        row = pd.DataFrame(
            [
                {
                    "rsi": rsi,
                    "ma_gap_pct": ma_gap,
                    "volatility_pct": vol,
                    "volume_change_pct": vchg,
                    "sentiment_score": sent,
                    "risk_per_trade_pct": risk,
                    "holding_days": hold,
                    "asset": asset,
                    "market_condition": market,
                    "experience_level": exp,
                }
            ]
        )
        p = model.predict_proba(row)[0, 1]
        if p >= 0.5:
            st.success(f"Predicted: **{POSITIVE}** (probability {p:.1%})")
        else:
            st.error(f"Predicted: **{NEGATIVE}** (probability of profit {p:.1%})")
        st.progress(float(p))
        st.caption("Fictional result for learning only. Not financial advice.")

# ---- Tab 4: Model results --------------------------------------------------
with tab4:
    st.header("Model results")
    cols = st.columns(len(metrics))
    for cc, (name, val) in zip(cols, metrics.items()):
        cc.metric(name, f"{val:.3f}")

    left, right = st.columns(2)
    with left:
        st.subheader("Confusion matrix")
        fig3, ax3 = plt.subplots(figsize=(4.5, 3.8))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=[NEGATIVE, POSITIVE],
            yticklabels=[NEGATIVE, POSITIVE],
            ax=ax3,
        )
        ax3.set_xlabel("Predicted")
        ax3.set_ylabel("Actual")
        st.pyplot(fig3)
        plt.close(fig3)
    with right:
        st.subheader("Classification report")
        st.code(report)

    st.subheader("Simulated portfolio (test trades only)")
    s1, s2 = st.columns(2)
    capital = s1.number_input("Starting capital (fictional $)", 1000, 1_000_000, 10_000, step=1000)
    pos = s2.slider("Share of capital used per trade (%)", 1, 50, 10)

    test_returns = df.loc[X_test.index, "trade_return_pct"]
    approved = test_returns[pred == 1]
    eq_all = simulate_portfolio(test_returns, capital, pos)
    eq_model = simulate_portfolio(approved, capital, pos)

    fig4, ax4 = plt.subplots(figsize=(9, 4))
    ax4.plot(eq_all, label=f"Take every trade ({len(test_returns)} trades)")
    ax4.plot(eq_model, label=f"Only model-approved trades ({len(approved)} trades)")
    ax4.set_xlabel("Trade number")
    ax4.set_ylabel("Portfolio value ($)")
    ax4.legend()
    st.pyplot(fig4)
    plt.close(fig4)

    m1, m2 = st.columns(2)
    m1.metric("Final value - all trades", f"${eq_all[-1]:,.0f}", f"{eq_all[-1] / capital - 1:.1%}")
    m2.metric("Final value - model-approved", f"${eq_model[-1]:,.0f}", f"{eq_model[-1] / capital - 1:.1%}")
    st.caption(
        "Both curves use fictional returns generated by the same synthetic rule the model "
        "learned, so the model looks good by design. Real markets are far less predictable."
    )

st.divider()
st.caption(
    "Student learning project. Educational demonstration only; not financial advice. "
    "Trading real money involves risk of loss."
)
