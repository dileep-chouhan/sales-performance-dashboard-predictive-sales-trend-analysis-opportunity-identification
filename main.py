import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
# --- 1. Synthetic Data Generation ---
np.random.seed(42) # for reproducibility
num_days = 365
dates = pd.date_range(start='2022-01-01', periods=num_days)
sales = 100 + 50 * np.sin(2 * np.pi * np.arange(num_days) / 30) + 20 * np.random.randn(num_days) #Seasonal trend with noise
promotions = np.random.choice([0,1], size=num_days, p=[0.9,0.1]) # 10% chance of a promotion each day
sales += promotions * 50 # Promotions boost sales
df = pd.DataFrame({'Date': dates, 'Sales': sales, 'Promotion': promotions})
df['Month'] = df['Date'].dt.month
df['Year'] = df['Date'].dt.year
# --- 2. Data Cleaning and Feature Engineering (Minimal in this synthetic example) ---
# No significant cleaning needed for this synthetic data.
# --- 3. Analysis ---
# Calculate monthly average sales
monthly_sales = df.groupby(['Year', 'Month'])['Sales'].mean().reset_index()
# Time series decomposition to separate trend, seasonality, and residuals
from statsmodels.tsa.seasonal import seasonal_decompose
decomposition = seasonal_decompose(df['Sales'], model='additive', period=30) # Assuming roughly 30-day seasonality
trend = decomposition.trend
seasonal = decomposition.seasonal
residual = decomposition.resid
#Simple prediction (using a rolling average for demonstration)
window = 30
df['Sales_Prediction'] = df['Sales'].rolling(window=window).mean()
#Identify potential high sales opportunities (days with sales significantly above the trend)
df['Sales_Above_Trend'] = df['Sales'] - trend
df['High_Potential'] = (df['Sales_Above_Trend'] > 2*df['Sales_Above_Trend'].std()) & (df['Promotion']==0) #Flag days with sales significantly above trend and no promotion
# --- 4. Visualization ---
#Plot 1: Monthly Average Sales
plt.figure(figsize=(12, 6))
sns.lineplot(x='Date', y='Sales', data=df, label='Actual Sales')
sns.lineplot(x='Date', y='Sales_Prediction', data=df, label='Predicted Sales')
plt.title('Monthly Average Sales Trend with Predictions')
plt.xlabel('Date')
plt.ylabel('Average Sales')
plt.legend()
plt.savefig('sales_trend_prediction.png')
print("Plot saved to sales_trend_prediction.png")
#Plot 2: Time Series Decomposition
plt.figure(figsize=(12, 8))
plt.subplot(411)
plt.plot(df['Sales'], label='Original')
plt.legend(loc='upper left')
plt.subplot(412)
plt.plot(trend, label='Trend')
plt.legend(loc='upper left')
plt.subplot(413)
plt.plot(seasonal, label='Seasonality')
plt.legend(loc='upper left')
plt.subplot(414)
plt.plot(residual, label='Residuals')
plt.legend(loc='upper left')
plt.tight_layout()
plt.savefig('time_series_decomposition.png')
print("Plot saved to time_series_decomposition.png")
#Plot 3: High Potential Sales Opportunities
plt.figure(figsize=(10,6))
plt.scatter(df['Date'], df['Sales'], c=df['High_Potential'], cmap='viridis', label='Sales')
plt.title('High Potential Sales Opportunities')
plt.xlabel('Date')
plt.ylabel('Sales')
plt.colorbar(label='High Potential (True/False)')
plt.savefig('high_potential_sales.png')
print("Plot saved to high_potential_sales.png")