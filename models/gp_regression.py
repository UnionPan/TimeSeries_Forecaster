import torch
import gpytorch
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import streamlit as st
import altair as alt
from datetime import datetime, timedelta
import os
from pathlib import Path



class GPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(GPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.LinearMean(input_size=1)
        self.covar_module = gpytorch.kernels.ConstantKernel()  + gpytorch.kernels.PeriodicKernel() + gpytorch.kernels.RBFKernel()
        #print(self.covar_module.parameters)
        # self.covar_module =  gpytorch.kernels.PeriodicKernel() + gpytorch.kernels.ConstantKernel() + gpytorch.kernels.MaternKernel()
    
    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x) # type: ignore


class GPRegressor:
    def __init__(self, data):
        self.likelihood = gpytorch.likelihoods.GaussianLikelihood()
        self.data = data
        self.data['Date'] = pd.to_datetime(self.data['# Date'])
        self.data['t'] = (self.data['Date'] - self.data['Date'].min()).dt.days.astype(float)
        

        self.train_y = 1e-5*torch.tensor(self.data['Receipt_Count'].values, dtype=torch.float32)
        self.train_x = torch.tensor(self.data['t'].values, dtype=torch.float32).unsqueeze(-1)
        
        self.model = GPModel(self.train_x, self.train_y, self.likelihood)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr= 0.1)
        self.mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.model)
        self.training_iter = 1000
        self.seed = 1024
        torch.manual_seed(self.seed)
        

    def train(self):
        # train the model using Adam
        self.model.train()
        self.likelihood.train()
        # print(self.train_x)
        # print(self.train_y)
        for i in range(self.training_iter):
            self.optimizer.zero_grad()
            output = self.model(self.train_x)
            loss =  - self.mll(output, self.train_y) # type: ignore
            loss.backward()
            if (i+1) % 100 == 0 or i == 0:
                print(f'Iteration {i + 1}/{self.training_iter} - Loss: {loss.item():.3f}')
            self.optimizer.step()


    def forecast(self, future_day):
        
        # Evaluation Mode
        self.model.eval()
        self.likelihood.eval()

        # Define Future Dates for Prediction
        
        last_date = self.data['t'].iloc[-1]
        future_days = [last_date + i for i in range(1, future_day + 1)]
        future_days_tensor = torch.tensor([ d for d in future_days], dtype=torch.float32).unsqueeze(-1)

        
        # future_dates = [last_date + timedelta(days=i) for i in range(1, future_days + 1)]
        # future_dates_ordinal = torch.tensor([d.toordinal() for d in future_dates], dtype=torch.float32).unsqueeze(-1)

        past_dates = self.data['t'].iloc[:]
        past_days_tensor = torch.tensor([ d for d in past_dates], dtype=torch.float32).unsqueeze(-1)
       # past_dates_ordinal = torch.tensor([d.toordinal() for d in past_dates], dtype=torch.float32).unsqueeze(-1)
        #print(past_days_tensor, future_days_tensor)
        # Forecasting
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            predictions = self.likelihood(self.model(future_days_tensor))
            fitted = self.likelihood(self.model(past_days_tensor))
            mean = predictions.mean
            mean_fit = fitted.mean
            lower, upper = predictions.confidence_region()
            lower_fit, upper_fit = fitted.confidence_region()
            variance = predictions.variance
            variance_fit = fitted.variance

        return mean, variance, mean_fit, variance_fit, upper, lower, upper_fit, lower_fit



    def save_model(self, filepath: str):
        """
        Saves the model and likelihood state dictionaries to the specified filepath.
        """
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'likelihood_state_dict': self.likelihood.state_dict()
        }, filepath)
        print(f"Model saved to {filepath}")


    def load_model(self, filepath: str):
        """
        Loads the model and likelihood state dictionaries from the specified filepath.
        
        """
        # Load state dictionaries
        checkpoint = torch.load(filepath)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.likelihood.load_state_dict(checkpoint['likelihood_state_dict'])
        
        self.model.eval()
        self.likelihood.eval()
        

def plot_with_altair(df_samples_2021, future_dates, pred_mean, pred_upper, pred_lower, y_min=None, y_max=None):
    
    # Ensure the future dates are combined into a single DataFrame
    future_df = pd.DataFrame({
        'Date': pd.to_datetime(future_dates),
        'Predicted Mean': 1e5 * pred_mean[len(df_samples_2021):],
        'Predicted Upper': 1e5 * pred_upper[len(df_samples_2021):],
        'Predicted Lower': 1e5 * pred_lower[len(df_samples_2021):]
    })
    
    # Add predicted means and confidence intervals for 2021 as well
    df_samples_2021['Predicted Mean'] = 1e5 * pred_mean[:len(df_samples_2021)]
    df_samples_2021['Predicted Upper'] = 1e5 * pred_upper[:len(df_samples_2021)]
    df_samples_2021['Predicted Lower'] = 1e5 * pred_lower[:len(df_samples_2021)]
    
    # Combine 2021 samples and future predictions
    combined_df = pd.concat([df_samples_2021, future_df])

    # Base chart for the predicted mean
    base = alt.Chart(combined_df).encode(
        x='Date:T'
    )
    
    # Zooming in the Y axis by specifying y_min and y_max
    y_scale = alt.Scale(domain=[y_min, y_max]) if y_min is not None and y_max is not None else alt.Scale()

    # Line for predicted mean
    pred_mean_line = base.mark_line(color='red').encode(
        y=alt.Y('Predicted Mean:Q', scale=y_scale)  # Apply y-axis zoom here
    )

    # Confidence interval band
    confidence_band = base.mark_area(opacity=0.3).encode(
        y=alt.Y('Predicted Lower:Q', scale=y_scale),  # Apply y-axis zoom here
        y2='Predicted Upper:Q'
    )

    # Points for actual samples
    actual_points = alt.Chart(df_samples_2021).mark_point(color='blue').encode(
        x='Date:T',
        y=alt.Y('Receipt_Count:Q', scale=y_scale)  # Apply y-axis zoom here
    )

    # Combine the charts
    chart = (actual_points + pred_mean_line + confidence_band).properties(
        title="Sample Data and Predictions with Confidence Intervals",
        width=800,
        height=400
    )

    # Display in Streamlit
    st.altair_chart(chart, use_container_width=True)


if __name__ == "__main__":

    pth = Path(os.getcwd())
    data = pd.read_csv(os.path.join(pth.parent, 'data/data_daily.csv'))

    # Data Preprocessing
    
    regressor = GPRegressor(data)
    regressor.load_model(os.path.join(pth.parent, 'data/saved_model'))
    #regressor.train()
    
    future_days = 100
    mean, variance, mean_fit, variance_fit, upper, lower, upper_fit, lower_fit = regressor.forecast(future_days)
    pred_mean = np.concatenate((mean_fit.numpy(), mean.numpy()))
    pred_upper =  np.concatenate((upper_fit.numpy(), upper.numpy()))
    pred_lower = np.concatenate((lower_fit.numpy(), lower.numpy()))
    
    regressor.save_model(os.path.join(pth.parent, 'data/saved_model'))

    last_date = data['Date'].iloc[-1]
    future_dates = [last_date + timedelta(days=i) for i in range(1, future_days + 1)]
    
    past_dates = data['Date'].iloc[:]
    
    # Plotting
    plt.figure(figsize=(12, 6))
    plt.plot(data['Date'], data['Receipt_Count'], '-*', label='Observed Data')
    plt.plot()
    plt.plot(past_dates, 1e5*mean_fit.numpy(), 'r', label='fitted mean')
    plt.fill_between(past_dates, 1e5*upper_fit.numpy(),  1e5*lower_fit.numpy(), alpha=0.5, label='Variance (fit)')
    plt.plot(future_dates, 1e5*mean.numpy(), 'b', label='Predictive Mean')
    plt.fill_between(future_dates,  1e5*upper.numpy(),  1e5*lower.numpy(), alpha=0.3, label='Variance (predict)')
    plt.xlabel('Date')
    plt.ylabel('Count')
    plt.legend()
    plt.title('Gaussian Process Regression for Time Series Forecasting')
    plt.show()
