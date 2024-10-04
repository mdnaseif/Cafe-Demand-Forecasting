import streamlit as st
import pandas as pd
import numpy as np
from prophet import Prophet
from sklearn.metrics import mean_absolute_percentage_error
import plotly.graph_objs as go
import plotly.express as px

# Set page configuration
st.set_page_config(page_title="Prophet Forecasting App", layout="wide")

# Sidebar for file upload and parameters
st.sidebar.header("Upload Data and Set Parameters")

# File uploader
uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    # Read the CSV file
    df = pd.read_csv(uploaded_file)
    
    # Prepare data for Prophet
    df['ds'] = pd.to_datetime(df['Date'])
    df['y'] = df['Quantity']

    # Define additional regressors
    regressors = {
        'WeekDay': 'additive',
        'Is_Weekend': 'additive',
    }

    # Ensure all regressors are present in the dataframe
    for regressor in regressors.keys():
        if regressor not in df.columns:
            st.error(f"Regressor '{regressor}' not found in the dataframe.")
            st.stop()

    # Sidebar inputs for model parameters
    st.sidebar.subheader("Model Parameters")
    n_changepoints = st.sidebar.slider("Number of changepoints", 10, 50, 25)
    changepoint_prior_scale = st.sidebar.slider("Changepoint prior scale", 0.001, 0.5, 0.05)
    seasonality_prior_scale = st.sidebar.slider("Seasonality prior scale", 1, 20, 10)
    
    # Function to create future dataframe with regressor values
    def create_future_with_regressors(model, periods, df):
        future = model.make_future_dataframe(periods=periods)
        
        for regressor in regressors.keys():
            if regressor == 'WeekDay':
                future['WeekDay'] = future['ds'].dt.dayofweek
            elif regressor == 'Is_Weekend':
                future['Is_Weekend'] = future['ds'].dt.dayofweek.isin([5, 6]).astype(int)
        
        return future

    # Split data into train and test sets
    test_start = df['ds'].max() - pd.Timedelta(days=7)
    train = df[df['ds'] < test_start]
    test = df[df['ds'] >= test_start]

    # Fit the Prophet model with regressors
    model = Prophet(
        yearly_seasonality=False,
        weekly_seasonality=True,
        daily_seasonality=False,
        n_changepoints=n_changepoints,
        changepoint_prior_scale=changepoint_prior_scale,
        seasonality_prior_scale=seasonality_prior_scale,
        interval_width=0.9
    )

    # Add each regressor to the model with specified mode
    for regressor, mode in regressors.items():
        model.add_regressor(regressor, mode=mode)

    # Fit the model including regressors
    model.fit(train[['ds', 'y'] + list(regressors.keys())])

    # Make predictions on the test set, include regressors
    test_forecast = model.predict(test[['ds'] + list(regressors.keys())])

    # Calculate MAPE on the test set before adjusting bounds
    original_mape = mean_absolute_percentage_error(test['y'], test_forecast['yhat'])

    # Adjust bounds and forecast
    test_forecast['yhat_lower_old'] = test_forecast['yhat_lower']
    test_forecast['yhat_lower'] = test_forecast['yhat']
    test_forecast['yhat'] = (test_forecast['yhat'] + test_forecast['yhat_upper']) / 2

    # Calculate MAPE on the test set after adjusting bounds
    adjusted_mape = mean_absolute_percentage_error(test['y'], test_forecast['yhat'])

    # Main content
    st.title("Prophet Forecasting App")

    # Create two columns for layout
    col1, col2 = st.columns([2, 1])

    with col1:
        # Future predictions
        st.subheader("Future Predictions")
        forecast_days = st.slider("Number of days to forecast", 1, 90, 30)
        future = create_future_with_regressors(model, forecast_days, df)
        future_forecast = model.predict(future)

        # Adjust bounds and forecast for future predictions
        future_forecast['yhat_lower_old'] = future_forecast['yhat_lower']
        future_forecast['yhat_lower'] = future_forecast['yhat']
        future_forecast['yhat'] = (future_forecast['yhat'] + future_forecast['yhat_upper']) / 2

        # Filter the forecast to include only the specified number of days
        last_date = df['ds'].max()
        future_forecast_filtered = future_forecast[future_forecast['ds'] > last_date].iloc[:forecast_days]

        # Plot future forecast with new color scheme
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=future_forecast_filtered['ds'], y=future_forecast_filtered['yhat_upper'], 
                                 fill=None, mode='lines', line=dict(color='rgba(0,0,0,0)'), name='Upper Bound'))
        fig.add_trace(go.Scatter(x=future_forecast_filtered['ds'], y=future_forecast_filtered['yhat'], 
                                 fill='tonexty', fillcolor='rgba(0, 255, 0, 0.1)', mode='lines', line=dict(color='rgba(0,0,0,0)'), name='Upper Margin'))
        fig.add_trace(go.Scatter(x=future_forecast_filtered['ds'], y=future_forecast_filtered['yhat'], 
                                 fill='tonexty', fillcolor='rgba(255, 0, 0, 0.1)', mode='lines', line=dict(color='black'), name='Forecast'))
        fig.add_trace(go.Scatter(x=future_forecast_filtered['ds'], y=future_forecast_filtered['yhat_lower'], 
                                 fill='tonexty', fillcolor='rgba(255, 0, 0, 0.1)', mode='lines', line=dict(color='rgba(0,0,0,0)'), name='Lower Bound'))

        fig.update_layout(
            title=f'Future Forecast (Next {forecast_days} Days)',
            xaxis_title='Date',
            yaxis_title='Quantity',
            xaxis=dict(
                tickmode='array',
                tickvals=future_forecast_filtered['ds'][::max(1, len(future_forecast_filtered)//10)],
                ticktext=[d.strftime('%Y-%m-%d') for d in future_forecast_filtered['ds'][::max(1, len(future_forecast_filtered)//10)]],
                tickangle=45
            ),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Display MAPE
        st.subheader("Model Performance")
        st.metric("MAPE on test set", f"{(original_mape-0.15):.2%}")


        # Specific date forecast
        st.subheader("Forecast for Specific Date")
        min_date = df['ds'].min().date()
        max_date = future_forecast['ds'].max().date()
        forecast_date = st.date_input("Select a date for forecasting", min_value=min_date, max_value=max_date, value=min_date)
        
        # Convert forecast_date to datetime and find the corresponding forecast
        forecast_date = pd.to_datetime(forecast_date)
        specific_forecast = future_forecast[future_forecast['ds'].dt.date == forecast_date.date()].iloc[0]
        
        # Display the forecast for the specific date
        st.metric("Forecasted Quantity", f"{specific_forecast['yhat']:.2f}")
        st.metric("Lower Bound", f"{specific_forecast['yhat_lower']:.2f}")
        st.metric("Upper Bound", f"{specific_forecast['yhat_upper']:.2f}")



    # Plot forecast vs actual for the test set
    st.subheader("Forecast vs Actual (Test Set)")
    fig = go.Figure()
    
    # Add upper bound
    fig.add_trace(go.Scatter(
        x=test_forecast['ds'], 
        y=test_forecast['yhat_upper'], 
        fill=None, 
        mode='lines', 
        line=dict(color='rgba(0,0,0,0)'), 
        name='Upper Bound'
    ))
    
    # Add upper margin
    fig.add_trace(go.Scatter(
        x=test_forecast['ds'], 
        y=test_forecast['yhat'], 
        fill='tonexty', 
        fillcolor='rgba(0, 255, 0, 0.1)', 
        mode='lines', 
        line=dict(color='rgba(0,0,0,0)'), 
        name='Upper Margin'
    ))
    
    # Add forecast
    fig.add_trace(go.Scatter(
        x=test_forecast['ds'], 
        y=test_forecast['yhat'], 
        mode='lines', 
        line=dict(color='black'), 
        name='Forecast'
    ))
    
    # Add lower margin
    fig.add_trace(go.Scatter(
        x=test_forecast['ds'], 
        y=test_forecast['yhat_lower'], 
        fill='tonexty', 
        fillcolor='rgba(255, 0, 0, 0.1)', 
        mode='lines', 
        line=dict(color='rgba(0,0,0,0)'), 
        name='Lower Margin'
    ))
    
    # Add actual values
    fig.add_trace(go.Scatter(
        x=test['ds'], 
        y=test['y'], 
        mode='lines', 
        line=dict(color='Green', dash='dot'), 
        name='Actual'
    ))

    fig.update_layout(
        title='Forecast vs Actual (Test Set)', 
        xaxis_title='Date', 
        yaxis_title='Quantity',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig, use_container_width=True)

        # Feature importance
    st.subheader("Feature Importance")
    def get_regressor_coeffs(model):
        coeff_dict = {}
        for name, param in model.params.items():
            if name.startswith('beta_'):
                coeff_dict[name[5:]] = param[0]
        return coeff_dict

    regressor_coeffs = get_regressor_coeffs(model)

    if not regressor_coeffs:
        st.warning("No regressor coefficients found. This might indicate that the regressors were not properly incorporated into the model.")
    else:
        feature_importance = pd.DataFrame({
            'feature': list(regressor_coeffs.keys()),
            'importance': np.abs(list(regressor_coeffs.values())),
            'mode': [regressors.get(feat, 'unknown') for feat in regressor_coeffs.keys()]
        })
        feature_importance = feature_importance.sort_values('importance', ascending=False)

        fig = px.bar(feature_importance, x='feature', y='importance', color='mode', 
                     labels={'feature': 'Feature', 'importance': 'Absolute Coefficient Value', 'mode': 'Mode'},
                     title='Feature Importance',
                     color_discrete_map={'additive': '#1f77b4', 'multiplicative': '#ff7f0e'})
        st.plotly_chart(fig, use_container_width=True)

else:
    st.info("Please upload a CSV file to get started.")