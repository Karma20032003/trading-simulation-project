# Simple Trading Simulation

## A Basic Machine-Learning Application Using Streamlit

This student project demonstrates a complete beginner-friendly machine-learning workflow using Python, scikit-learn, and Streamlit. The application generates a **synthetic dataset of 800 fictional trades**, trains a logistic-regression model, presents exploratory charts, provides an interactive prediction demonstration, and runs a simple simulated portfolio.

> **Important:** This project uses synthetic data only. It does not contain real market data and must not be used as financial advice, an investment tool, or a real trading system.

## Project Objective

The project investigates the following research question:

> Can basic technical, sentiment, and risk variables predict a synthetically generated "Profitable trade" label?

The purpose is to teach students how to:

- Generate and explore a dataset
- Prepare numeric and categorical variables
- Divide data into training and testing sets
- Train a classification model
- Evaluate model performance
- Build an interactive Streamlit interface
- Simulate a simple portfolio from model signals
- Discuss limitations and responsible AI use

## Project Structure

```
trading-simulation-streamlit/
├── app.py
├── requirements.txt
└── README.md
```

The entire application, dataset-generation process, machine-learning pipeline, and portfolio simulation are contained in `app.py`. No separate dataset or model file is required.

## Dataset

The application automatically generates 800 fictional trade records. The variables include:

| Variable | Description |
| --- | --- |
| `asset` | Fictional asset (ALPHA, BETA, GAMMA, DELTA, OMEGA) |
| `market_condition` | Bullish, Sideways, or Bearish |
| `experience_level` | Beginner, Intermediate, or Advanced trader |
| `rsi` | Relative Strength Index (10-90) |
| `ma_gap_pct` | Short vs long moving-average gap (%) |
| `volatility_pct` | Price volatility (%) |
| `volume_change_pct` | Change in trading volume (%) |
| `sentiment_score` | News-sentiment score from 1 to 10 |
| `risk_per_trade_pct` | Share of capital risked per trade (%) |
| `holding_days` | Days the trade is held |
| `trade_return_pct` | Simulated percentage return (used by the portfolio simulation) |
| `trade_outcome` | Synthetic target variable |

The target has two classes:

- `Profitable trade`
- `Loss-making trade`

The target is created through a transparent fictional rule plus random noise (a trade is profitable when its simulated return is above zero). Therefore, the results cannot be generalized to real markets.

## Machine-Learning Method

The application uses **logistic regression** because it is suitable for basic binary classification and is easier to explain than many complex models.

The workflow is:

1. Generate reproducible synthetic data using random seed 42.
2. Separate the input variables and target variable.
3. Divide the dataset into 80% training and 20% testing data.
4. Replace missing numeric values with the median.
5. Replace missing categorical values with the most frequent category.
6. Standardize numeric variables.
7. One-hot encode categorical variables.
8. Train a logistic-regression classifier.
9. Evaluate the model using unseen testing data.
10. Simulate a portfolio on the test trades.

## Evaluation Measures

The application presents:

- Accuracy
- ROC-AUC
- Precision
- Recall
- F1-score
- Confusion matrix
- Classification report

Using the fixed synthetic dataset and test split, the expected results are approximately:

- **Accuracy: 73.1%**
- **ROC-AUC: 0.802**

These values show how the model performs on its generated test data. They do not demonstrate real-world trading skill.

## Portfolio Simulation

The **Model results** tab compares two fictional portfolios built from the test trades:

1. **Take every trade**
2. **Only model-approved trades** (trades predicted as profitable)

You can change the starting capital and the share of capital used per trade. Capital compounds after every trade. Because the returns follow the same synthetic rule the model learned, the model looks good by design.

## Streamlit Pages

The application contains four tabs:

1. **Overview** - explains the project, model, and responsible-use conditions.
2. **Explore data** - displays a selectable distribution chart, a correlation heatmap, and fictional records.
3. **Prediction demo** - allows users to enter one fictional trade setup.
4. **Model results** - presents evaluation metrics, the confusion matrix, and the portfolio simulation.

## Installation and Local Use

### 1. Download or clone the repository

```
git clone https://github.com/YOUR-USERNAME/trading-simulation-streamlit.git
cd trading-simulation-streamlit
```

Replace `YOUR-USERNAME` with your GitHub username.

### 2. Create a virtual environment

Windows:

```
python -m venv .venv
.venv\Scripts\activate
```

macOS or Linux:

```
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install the packages

```
pip install -r requirements.txt
```

### 4. Run the application

```
streamlit run app.py
```

The application should open at `http://localhost:8501`.

## requirements.txt

```
streamlit
pandas
numpy
scikit-learn
matplotlib
seaborn
```

## Deploy on Streamlit Community Cloud

1. Create a new GitHub repository.
2. Upload `app.py`, `requirements.txt`, and `README.md`.
3. Sign in to [Streamlit Community Cloud](https://share.streamlit.io/).
4. Select **Create app**.
5. Select the GitHub repository and branch.
6. Enter `app.py` as the main file path.
7. Select **Deploy**.
8. Open the deployed application and test every tab.

## Ethical Considerations

Trading and investment decisions affect real money. A real project would require:

- Clear disclosure that predictions are not financial advice
- Use of licensed, accurate market data
- Realistic costs (fees, spread, slippage, taxes)
- Out-of-sample and walk-forward testing to avoid overfitting
- Risk controls and loss limits
- Compliance with local financial regulations
- Transparent communication of the risk of loss

The model must not be used to make real investment decisions.

## Limitations

- The dataset is artificial rather than collected from real markets.
- The target is generated from a known rule, not from real price movements.
- The simulation ignores fees, slippage, taxes, and liquidity.
- Logistic regression may not capture complex market relationships.
- The application has not been tested on real data.
- Results cannot be interpreted as findings about real markets.
- Strong performance is partly expected because the target follows a known synthetic rule.

## Future Improvements

- Use real historical price data with proper licensing
- Compare multiple algorithms (random forest, gradient boosting)
- Use time-series cross-validation instead of a random split
- Add transaction costs and stop-loss rules
- Add model-explanation techniques
- Add risk metrics such as drawdown and Sharpe ratio

## Suggested Screenshots for Submission

1. Application overview
2. Data-exploration chart
3. Fictional-trade prediction
4. Accuracy and ROC-AUC
5. Confusion matrix
6. Classification report
7. Portfolio simulation chart

## Suggested Questions to Answer

1. Why was the model (logistic regression) selected?
2. Why is the dataset divided into training and testing sets?
3. What does one-hot encoding do?
4. Why are numeric variables standardized?
5. What is the difference between accuracy and recall?
6. What does ROC-AUC measure?
7. Why can this application not be used for real trading?
8. What are the advantages and limitations of synthetic data?
9. Why does the portfolio simulation look better than real trading would?
10. How could the model be improved in future research?

## Disclaimer

This repository is a student learning project. The application and its predictions are for educational demonstration only and are not financial advice. Trading involves risk of loss.

## Author

**Student name:** Add your name  
**Institution:** Add your institution  
**Course:** Add your course  
**Year:** 2026
