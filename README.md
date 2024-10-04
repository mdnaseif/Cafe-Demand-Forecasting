# Cafe Demand Forecasting

## Project Overview
This project implements a daily demand forecasting system for a cafe to address issues of stock management and optimize inventory. By predicting product demand, the cafe aims to reduce instances of out-of-stock and over-stock situations, thereby minimizing wastage costs and maximizing profit.

## Problem Statement
The cafe is experiencing daily fluctuations in stock levels, leading to:
1. Out-of-stock situations
2. Over-stock situations

These issues result in increased wastage costs and decreased profits. The goal is to develop a daily demand forecasting model for the cafe's products to improve inventory management.

## Data Collection
- Data source: POS (Point of Sale) system
- Collection method: Web scraping using Octoparse
- Dataset: Approximately 100,000 customer bills (receipts)

## Data Preparation and Processing
1. Data cleaning: Removal of unused columns, handling duplicates and null values
2. Date/time formatting: Converting date strings to datetime objects
3. Feature engineering: 
   - Extracting year, month, day, weekday
   - Creating weekend indicator
   - Calculating daily sales aggregates

## Exploratory Data Analysis
The project includes various visualizations to understand sales patterns:
1. Daily sales trend
2. Monthly sales comparison
3. Weekday vs. weekend sales
4. Quantity vs. net sales scatter plot
5. Gross profit margin over time
6. Top 10 days by sales

## Forecasting Model
The project uses Facebook's Prophet library for time series forecasting.

Key features:
- Incorporation of additional regressors (WeekDay, Is_Weekend)
- Model parameters tuning (changepoints, seasonality)
- Train-test split for model evaluation
- RMSE (Root Mean Square Error) calculation for model performance assessment

## Files in the Repository
1. `app.py`: Streamlit web application for interactive forecasting
2. `main.ipynb`: Jupyter notebook containing data preparation, analysis, and modeling steps

## How to Use
1. Install required dependencies:
   ```
   pip install streamlit pandas numpy prophet sklearn plotly
   ```
2. Run the Streamlit app:
   ```
   streamlit run app.py
   ```
3. Upload your CSV file containing sales data
4. Adjust model parameters using the sidebar
5. View forecasts and performance metrics

## Future Improvements
- Incorporate more external factors (e.g., weather, local events)
- Implement automated data collection from the POS system
- Develop a more sophisticated demand categorization system
- Integrate with inventory management system for real-time recommendations

## Contributors
[MD NASEIF]

## License
[MIT]
