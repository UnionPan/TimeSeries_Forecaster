# TimeSeries_Forecaster
Receipts forecaster for Fetch


## Setting Up the Virtual Environment

To ensure all dependencies are isolated, follow these steps to set up a virtual environment using conda. 

### Using Conda

1. **Install Conda (If Not Already Installed):**

   If you don't have Conda installed, you can download and install [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or [Anaconda](https://www.anaconda.com/products/distribution).

2. **Create a Conda Environment:**

   Create a new Conda environment named `forecaster_env` with Python 3.9.

   ```bash
   conda create --name forecaster_env python=3.9


4. **Install Dependencies:**

    ```bash
    pip install -r requirements.txt
    ```


## Running the Application

After setting up the environment and installing dependencies, you can launch the Streamlit application.

```bash
streamlit run forecaster.py