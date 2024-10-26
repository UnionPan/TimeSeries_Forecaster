import streamlit as st
import pandas as pd
import numpy as np
import time
import torch
import altair as alt
from datetime import datetime, date
from models.monte_carlo import linear_forecaster, OLS, plot_monte_carlo_trajectories
from models.gp_regression import GPRegressor, GPModel, plot_with_altair
import gpytorch
import os
from pathlib import Path

# Main index file, we set the title of the app as
st.title('Daily Receipts Scanning Forecaster')

# File uploader widget
uploaded_file = st.file_uploader(
    'Upload the history receipt scanning counts file (CSV) here', type=['csv']
)

if uploaded_file is not None:
    # Add a button to rerun the data loading and plotting
    rerun_button = st.button("Reload")
    
    # Check if the data has already been processed or if rerun_button is clicked
    if 'data_loaded' not in st.session_state or rerun_button:
        # Read the CSV file
        data = pd.read_csv(uploaded_file)
        
        # Convert date column to datetime format
        data['Date'] = pd.to_datetime(data["# Date"])
        data['Count'] = data["Receipt_Count"]
        
        # Ordinary Least Square Estimate
        beta, gamma, sigma = OLS(data)
        
        N = len(data)
        
        # Initialize the progress bar
        progress_bar = st.sidebar.progress(0)
        status_text = st.sidebar.empty()
        
        # Initialize an empty DataFrame to store the data for plotting
        chart_data = pd.DataFrame(columns=['Date', 'Count'])
        
        # Placeholder for the chart
        chart_placeholder = st.empty()
        
        # Loop over the data to simulate progress
        chunk_size = 5  # Number of data points to add in each iteration
        for i in range(0, N, chunk_size):
            # Get the next chunk of data
            new_data = data.iloc[i:i + chunk_size]
            
            start_date = new_data['Date'].iloc[0].strftime('%Y-%m-%d')
            end_date = new_data['Date'].iloc[-1].strftime('%Y-%m-%d')
            status_text.text(f"Adding data from {start_date} to {end_date}")
            
            chart_data = pd.concat(
                [chart_data, new_data[['Date', 'Count']]], ignore_index=True
            )
            
            # Create time variable 't' as the number of days since the first date
            chart_data['t'] = (chart_data['Date'] - chart_data['Date'].min()).dt.days.astype(float)
            
            # Calculate fitted values
            chart_data['Fitted'] = beta + gamma * chart_data['t']
            
            # Calculate confidence intervals
            k = 1.96  # For a 95% confidence interval
            chart_data['Upper_Bound'] = chart_data['Fitted'] + k * sigma
            chart_data['Lower_Bound'] = chart_data['Fitted'] - k * sigma
            
            y_min = chart_data['Count'].min()
            y_max = chart_data['Count'].max()
            
            base = alt.Chart(chart_data).encode(x='Date:T')
            
            # Original data line
            observed_line = base.mark_line(color='blue').encode(
                y=alt.Y('Count', scale=alt.Scale(domain=[y_min - 500000, y_max + 500000]))
            )
            
            # Fitted trend line
            fitted_line = base.mark_line(color='red').encode(
                y=alt.Y('Fitted', scale=alt.Scale(domain=[y_min - 500000, y_max + 500000]))
            )
            
            # Confidence interval area
            confidence_interval = base.mark_area(color='lightcoral', opacity=0.1).encode(
                y=alt.Y('Lower_Bound', scale=alt.Scale(domain=[y_min - 500000, y_max + 500000])),
                y2='Upper_Bound'
            )
            
            # Combine the layers
            chart = (confidence_interval + observed_line + fitted_line).properties(
                width='container',
                height=400
            )
            
            
            # Display the chart
            chart_placeholder.altair_chart(chart, use_container_width=True)
            
            # Update the progress bar
            progress = int(100 * (i + chunk_size) / N)
            progress_bar.progress(min(progress, 100))
            
            # Sleep to simulate time delay (adjust as needed)
            time.sleep(0.01)
        
        # Update the status text after the loop is complete
        status_text.text("Completed plotting all data.")
        
        # # --- Time Series Analysis Section --- We perform two tests
        # st.write("Performing Mean-Reverting Test")
        
        # st.write("Performing Stationarity Test")
    
        
        # Clear the progress bar
        progress_bar.empty()
        
        # Store the processed data and chart data in session state
        st.session_state['data_loaded'] = True
        st.session_state['data'] = data
        st.session_state['chart_data'] = chart_data
        st.session_state['beta'] = beta
        st.session_state['gamma'] = gamma
        st.session_state['sigma'] = sigma

    else:
        # Data has already been loaded and processed
        data = st.session_state['data']
        chart_data = st.session_state['chart_data']
        beta = st.session_state['beta']
        gamma = st.session_state['gamma'] 
        sigma = st.session_state['sigma']
        
        # Determine y-axis limits to fit the data tightly
        y_min = chart_data['Count'].min()
        y_max = chart_data['Count'].max()
        
        # Display the chart without rerunning the progress bar
        chart = alt.Chart(chart_data).mark_line().encode(
            x='Date:T',
            y=alt.Y('Count', scale=alt.Scale(domain=[y_min - 500000, y_max + 500000]))
        ).properties(
            width='container',
            height=400
        )
        st.altair_chart(chart, use_container_width=True)
    
  
    
    # --- Forecasting Section ---
    st.write("Select a date in 2022 to forecast the receipt scanning counts.")
    
    # Define the date range for 2022 for the date input widget
    start_date_2022 = date(2022, 1, 1)
    end_date_2022 = date(2022, 12, 31)
    selected_date = st.date_input(
        'Select a date',
        value=start_date_2022,
        min_value=start_date_2022,
        max_value=end_date_2022
    )
    
    
    # When a date is selected, call the forecaster function and display the results
    if selected_date:
        # Call the forecaster function
        day_of_year = selected_date.timetuple().tm_yday
        # st.write("day of the year {}".format(day_of_year))
        bias = data['Count'][-1:].item()
        gamma = st.session_state['gamma']
        sigma = st.session_state['sigma']
        # if use_model:
        #     bias = beta + gamma * 365
        forecasted_ranges, forecasted_number, sim_traj = linear_forecaster(bias, gamma, sigma, day_of_year)
        
        # Display the forecasted ranges
        st.subheader("Ordinary Least Square + Monte-Carlo")
        for confidence_level, forecast_range in forecasted_ranges.items():
            st.write(f"{confidence_level} Confidence Interval: [ {forecast_range[0]},  {forecast_range[1]} ]")
        st.write(f"Forecasted receipt scanning for {selected_date.strftime('%Y-%m-%d')} is {forecasted_number}")
        
        start_date = start_date_2022
        
        plot_monte_carlo_trajectories(sim_traj, selected_date, start_date)

        # ---- Gaussian Process Regression ------
        
        st.write("Let's play around Non-parametric ones")
        
        pth = Path(os.getcwd())
        
        regressor = GPRegressor(data)
        #regressor.load_model(regressor.load_model(os.path.join(pth, 'data/saved_model')))
        if st.button("Run Gaussian Process Regression"):
            regressor.train()
            regressor.save_model(os.path.join(pth, 'data/saved_model'))
            st.write("Training complete!!")
        
        mean, variance, mean_fit, variance_fit, upper, lower, upper_fit, lower_fit = regressor.forecast(day_of_year)
        
        # Create time variable 't' as the number of days since the first date
        chart_data['t'] = (chart_data['Date'] - chart_data['Date'].min()).dt.days.astype(float)
        
        pred_mean = np.concatenate((mean_fit.numpy(), mean.numpy()))
        pred_upper =  np.concatenate((upper_fit.numpy(), upper.numpy()))
        pred_lower = np.concatenate((lower_fit.numpy(), lower.numpy()))
        future_dates = pd.date_range(start=start_date_2022, end=selected_date, freq='D')

        plot_with_altair(data, future_dates, pred_mean=pred_mean, pred_upper=pred_upper, pred_lower=pred_lower, y_min=y_min-500000, y_max=y_max+500000)
          

else:
    st.write('Upload your time series data to begin.')
