
# a Monte Carlo simulator for stochastic linear series
import torch
import numpy as np 
import pandas as pd
import streamlit as st
import altair as alt

def OLS(data):
    # given a time seris data with two columns date and count, using Ordinary Least Square to identify the system
    data['Date'] = pd.to_datetime(data['Date'])
    data['t'] = (data['Date'] - data['Date'].min()).dt.days.astype(float)
    t = torch.tensor(data['t'].values, dtype=torch.float32)
    y = torch.tensor(data['Count'].values, dtype=torch.float32)
    ones = torch.ones_like(t)
    X = torch.stack([ones, t], dim=1)
    XTX, XTy = X.t().mm(X), X.t().mv(y)
    
    vec_inv = torch.linalg.solve(XTX, XTy)
    beta = vec_inv[0].item()
    gamma = vec_inv[1].item()
    y_pred = X.mm(vec_inv.unsqueeze(1)).squeeze()
    residuals = y - y_pred
    sigma_squared = (residuals ** 2).sum() / (len(t) - 1)
    volatility = torch.sqrt(sigma_squared).item()
    
    return beta, gamma, volatility


def linear_forecaster(beta, gamma, volatility, day_of_year):
    # Convert selected_date to t, an integer between 1 and 365
    t = day_of_year

    num_simulations = 10000  # Number of trajectories to simulate

    # Initialize y0 = beta
    y_t = torch.full((num_simulations,), beta, dtype=torch.float32)
    
    sim_res = y_t.reshape(num_simulations,1)

    # Simulate up to time t
    
    for i in range(t):
        # Random volatility term (Gaussian noise with mean 0 and std = volatility)
        random_noise = torch.randn(num_simulations) * volatility
        # Update y_t
        y_t =  ( beta + gamma *i + random_noise).int()
        
        sim_res = torch.cat((sim_res, y_t.reshape(num_simulations,1)), dim=1)

    # Compute confidence intervals
    # Sort the final y_t values
    y_t_sorted, _ = torch.sort(y_t)

    forecast_ranges = {}

    # Confidence intervals
    # 99% confidence interval corresponds to percentiles 0.5% and 99.5%
    lower_idx_99 = int(0.005 * num_simulations)
    upper_idx_99 = int(0.995 * num_simulations) - 1

    # 95% confidence interval corresponds to percentiles 2.5% and 97.5%
    lower_idx_95 = int(0.025 * num_simulations)
    upper_idx_95 = int(0.975 * num_simulations) - 1

    # 90% confidence interval corresponds to percentiles 5% and 95%
    lower_idx_90 = int(0.05 * num_simulations)
    upper_idx_90 = int(0.95 * num_simulations) - 1

    forecast_ranges['99%'] = [int(y_t_sorted[lower_idx_99].item()), int(y_t_sorted[upper_idx_99].item())]
    forecast_ranges['95%'] = [int(y_t_sorted[lower_idx_95].item()), int(y_t_sorted[upper_idx_95].item())]
    forecast_ranges['90%'] = [int(y_t_sorted[lower_idx_90].item()), int(y_t_sorted[upper_idx_90].item())]
    
    forecast_number = torch.mean(y_t_sorted.float()).int()

    return forecast_ranges, forecast_number, sim_res

def plot_monte_carlo_trajectories(sim_res, selected_date, start_date):
    """
    Plots Monte Carlo simulation trajectories up to the selected date.

    Parameters:
    - sim_res: torch.Tensor of shape (num_simulations, t + 1)
    - selected_date: datetime.date object representing the selected date
    - start_date: datetime.date object representing the start date of simulations
    """
    # Convert sim_res to a pandas DataFrame
    num_simulations, num_time_steps = sim_res.shape
    time_steps = range(num_time_steps)  # Time steps from 0 to t

    # Create a DataFrame with columns as time steps and rows as simulations
    sim_df = pd.DataFrame(sim_res.numpy(), columns=time_steps)

    # Melt the DataFrame to long format for Altair
    sim_df_long = sim_df.reset_index().melt(id_vars='index', var_name='Time', value_name='Value')
    sim_df_long.rename(columns={'index': 'Simulation'}, inplace=True)

    # Convert time steps to dates
    sim_df_long['Date'] = sim_df_long['Time'].apply(lambda x: start_date + pd.Timedelta(days=x))

    # Calculate y-axis limits with dynamic padding
    y_min = sim_df_long['Value'].min()
    y_max = sim_df_long['Value'].max()
    data_range = y_max - y_min
    padding = data_range * 0.1  # 10% padding
    y_axis_domain = [y_min - padding, y_max + padding]

    # Plotting using Altair
    # Limit the number of trajectories plotted for performance (optional)
    max_trajectories_to_plot = 100  # Adjust as needed
    sampled_simulations = sim_df_long['Simulation'].unique()[:max_trajectories_to_plot]
    sim_df_plot = sim_df_long[sim_df_long['Simulation'].isin(sampled_simulations)]

    chart = alt.Chart(sim_df_plot).mark_line(opacity=0.2).encode(
        x='Date:T',
        y=alt.Y('Value:Q', scale=alt.Scale(domain=y_axis_domain)),
        color=alt.Color('Simulation:N', legend=None)
    ).properties(
        width='container',
        height=400,
        title="Monte Carlo Trajectories"
    )

    st.altair_chart(chart, use_container_width=True)

if __name__ == "__main__":
    0
   