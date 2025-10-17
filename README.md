# Demand Forecasting Project

This project predicts daily product demand at the store level using machine learning.  
It helps identify which stores and product categories are expected to sell more or less in the next period.

---

## Goal

The main goal was to build a forecasting model that can predict future sales based on historical data and external factors such as store info, promotions, oil prices, and holidays.

---

## Dataset

- Over 3 million rows of daily sales data  
- 54 stores and 33 product categories  
- Features include: `store_nbr`, `family`, `onpromotion`, `date`, `oil price`, `holiday flag`, lag features, and rolling averages.  

Large raw files (over 100MB) are not included due to GitHub limits.  
All scripts are available in the `/src` folder.

---

## Project Structure

---

## Model Training

- Model: LightGBM Regressor  
- Hyperparameter tuning with GridSearchCV  
- Metrics: RMSE, MAPE, R²  

Best model results:
- MAPE: ~17%  
- R²: 0.83  

---

## Results

**Top Performing Categories:**
| Category | Avg Predicted Sales |
|-----------|--------------------|
| GROCERY I | 142.0 |
| PRODUCE | 134.7 |
| BEVERAGES | 123.7 |
| CLEANING | 108.8 |
| DAIRY | 103.0 |

**Lowest Performing Categories:**
| Category | Avg Predicted Sales |
|-----------|--------------------|
| LADIESWEAR | 47.7 |
| LINGERIE | 47.7 |
| MAGAZINES | 47.5 |
| HARDWARE | 47.1 |
| HOME APPLIANCES | 46.4 |

---

## Power BI Dashboard

Power BI report visualizes store accuracy, forecast vs actual, and category performance.  
File: `Demand_Forecast_Dashboard.pbix`

Key Insights

The model predicts around 83% of store-level demand correctly.

Grocery and Produce categories dominate total sales.

Categories such as Magazines and Hardware show lower demand.

Helps improve inventory and marketing planning.

Future Improvements

Try other models such as LSTM, Prophet, or XGBoost.

Add weather and macroeconomic indicators.

Automate weekly forecast updates.