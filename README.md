# DSCI 352 Final Project: Churn Prediction & Crypto Forecasting

This repository contains the final project for DSCI 352, focusing on two distinct supervised learning challenges:

1. **Telco Customer Churn:** Binary classification on tabular data using Scikit-Learn and Keras.
2. **Bitcoin Price Forecasting:** Time-series regression using LSTM neural networks.
3. **Automated Pipeline:** A serverless deployment of the churn model using AWS Lambda and Streamlit.



## Project Structure

```
.
├── part1&part3/                     # Telco Churn Modeling & AWS Pipeline
│   ├── data/                        # Contains WA_Fn-UseC_-Telco-Customer-Churn.csv
│   ├── telco_churn_output/          # Generated artifacts (plots, JSONs, CSVs)
│   ├── part1_telco_churn.py         # Main training script
│   ├── part1_telco_churn_viz.py     # Visualization generator
│   ├── part3_telco_churn_dashboard.py  # Streamlit App
│   └── part3_telco_churn_lambda_function.py # Pure Python code for AWS Lambda
│
└── part2/                           # Bitcoin Price Forecasting
    ├── COINBASE_BTCUSD_1D.csv       # Historical Price Data
    └── Problem2.ipynb               # LSTM Modeling Notebook
```



## Dependencies

You can install all required packages for both parts using `pip`.

- Create a `requirements.txt` file:

  ```shell
  numpy>=1.21.0
  pandas>=1.3.0
  matplotlib>=3.4.0
  seaborn>=0.11.0
  tensorflow>=2.8.0
  scikit-learn>=1.0.0
  jupyter>=1.0.0
  boto3>=1.20.0
  streamlit>=1.10.0
  plotly>=5.0.0
  requests>=2.25.0
  ```

- Install: 

  ```shell
  pip install -r requirements.txt`
  ```



## System Requirements

- **Python:** **3.10** (Recommended to match AWS Lambda runtime and ensure TensorFlow compatibility)

- **RAM:** Minimum 4GB (8GB recommended for LSTM training)

- **Disk Space**: ~100MB for dependencies and data



------

## Part 1: Telco Customer Churn Modeling

This section builds and compares Logistic Regression, Random Forest, Gradient Boosting, and a Keras Neural Network to predict customer churn.

### Prerequisites

1. Ensure all dependencies are installed (see main `requirements.txt`).
2. Ensure the dataset `WA_Fn-UseC_-Telco-Customer-Churn.csv` is located in the `part1&part3/data/` directory (or update `LOCAL_CSV` path in the script)

### Script Workflow

The `part1_telco_churn.py` script follows this sequence:

1. **Data Loading & Cleaning**
   - Loads the dataset and fixes variable types (e.g., coercing `TotalCharges` to numeric).
   - Maps the target variable `Churn` to binary (0/1).
2. **Preprocessing Pipeline**
   - Numeric: Imputes missing values with the median and applies Standard Scaling.
   - Categorical: Imputes missing values with the most frequent and applies One-Hot Encoding.
3. **Model Training**
   - Scikit-Learn: Trains Logistic Regression, Random Forest, Gradient Boosting, and SGD Classifier.
   - Deep Learning: Trains a Keras MLP (Multi-Layer Perceptron) with Dropout layers for regularization.
4. **Gradient Stability Analysis**
   - Uses a custom `GradientTracker` callback to monitor gradient magnitudes during Keras training.
   - Flags potential "Vanishing" or "Exploding" gradient issues.

5. **Automated Model Selection Policy**: The script applies a specific logic to choose the final production model:
   - Primary Choice: It prefers the Keras Deep Learning model if it is competitive (within 0.01 AUC of the best Sklearn model) AND has healthy gradients (no vanishing/exploding).
   - Fallback: Otherwise, it defaults to the Scikit-Learn model with the highest validation AUC.

6. **Artifact Export (AWS Ready)**

   - With `ENABLE_AWS_EXPORT = True`, the script extracts the raw coefficients (weights) and preprocessing statistics of the selected model.

   - These are saved as a lightweight JSON artifact (removing the dependency on heavy libraries like Scikit-Learn for inference) and uploaded to S3.

### Running the Models

1. Navigate to the `part1` directory

   ```shell
   cd "part1&part3"
   ```

2. Running the Training Script

   ```shell
   python part1_telco_churn.py
   ```

   Output (Generated in the script folder):

   - `keras_history.csv`: Raw training metrics (Loss/AUC) for every epoch.
   - `keras_gradients.json`: Detailed gradient statistics used for stability analysis.
   - `model_leaderboard_telco.json`: The final performance scores of all trained models.
   - Uploads `models/telco_churn_light.json` to AWS S3.

3. Generate Visualizations: After training, run the visualization script to generate the report figures

   ```shell
   python part1_telco_churn_viz.py
   ```

   Output (Saved to `telco_churn_output/`):

   - `model_leaderboard_auc.png`: Bar chart comparing all models.
   - `gradient_stability_log_scale.png`: Analysis of deep learning stability.
   - `keras_training_history.png`: Learning curves.
   - `model_comparison_table.csv`: Final metrics table for the report.

### Expected Runtime

- **Data preprocessing**: ~1-2 seconds
- **Training**: ~1-2 minutes
- **Visualization**: ~5 seconds





---

## Part 2: Bitcoin Price Forecasting (LSTM)

This section uses a Long Short-Term Memory (LSTM) network to forecast daily Bitcoin high prices.

### Prerequisites

1. Ensure all dependencies are installed (see above)
2. Make sure `COINBASE_BTCUSD_1D.csv` is in the same directory as `Problem2.ipynb`

### Notebook Workflow

The notebook follows this sequence:

1. **Data Loading**: Loads Bitcoin price data from CSV
2. **Data Preprocessing**:
   - Renames 'time' column to 'Date'
   - Converts to datetime and sets as index
   - Keeps only the 'high' price column
   - Splits data into train (2019-2022) and test (2023+)
3. **Data Scaling**: Applies MinMaxScaler to normalize prices (0-1 range)
4. **Sequence Creation**: Creates 30-day sliding windows for LSTM input
5. **Model Building**: Constructs LSTM network with 2 layers (50 units each)
6. **Model Training**: Trains for 20 epochs with Adam optimizer
7. **Prediction**: Makes predictions on test data
8. **Evaluation**: Calculates RMSE and visualizes results

### Running the Model

1. Navigate to the `part2` directory:

   ```shell
   cd ../part2
   ```

2. Launch Jupyter Notebook:

   ```shell
   jupyter notebook Problem2.ipynb
   ```

3. Run all cells by clicking `Cell` → `Run All` in the menu or use the keyboard shortcut: `Shift + Enter` to run cells sequentially

   - Load `COINBASE_BTCUSD_1D.csv`.
   - Preprocess and scale data (MinMax scaling).
   - Train the LSTM network.
   - Visualize Actual vs. Predicted prices.

### Expected Runtime

- **Data preprocessing**: ~1-2 seconds
- **Model training**: ~2-5 minutes (depending on hardware)
- **Prediction**: ~10-30 seconds
- **Total**: ~3-6 minutes

### Notes

- The model uses temporal train/test split (no randomization) to preserve time series structure
- Training data: 2019-2022
- Test data: 2023 onwards
- All prices are scaled to 0-1 range before training



---

## Part 3: Automated AWS Pipeline

This section launches a local dashboard that connects to an AWS Lambda function for real-time inference.

### Architecture

The pipeline follows a serverless microservices pattern:

- **Frontend:** A Streamlit web application for user interaction.
- **Backend:** An AWS Lambda function acting as the inference engine. It uses a "Pure Python" implementation (no heavy libraries like Scikit-Learn) to minimize cold-start latency.
- **API Layer:** Amazon API Gateway exposes the Lambda function as a public HTTP endpoint.
- **Model Registry:** **Amazon S3** (`dsci352-telco-churn-project`) stores the lightweight JSON model artifact.

 ### Prerequisites

1. Ensure you have installed the dashboard requirements: `streamlit`, `requests`, `plotly`, `pandas`
2. Internet Connection: The dashboard must connect to the live AWS API endpoint defined in `part3_telco_churn_dashboard.py`.

### Running the Dashboard

1. Navigate to the directory:

   ```shell
   cd "part1&part3"
   ```

2. Launch the Streamlit App:

   ```shell
   streamlit run part3_telco_churn_dashboard.py
   ```

3. Access the Interface: The dashboard will open automatically in your default web browser

### Dashboard Features

1. **Single Customer Analysis**
   - Input Form: Adjust demographics, services, and billing information using interactive sliders and dropdowns.
   - Real-time Inference: Clicking "Predict Churn Risk" sends a JSON payload to AWS Lambda.
   - Visualization: Displays a gauge chart showing the exact probability and a risk assessment (Loyal vs. High Risk).

2. **Batch Prediction Dashboard**

   - Bulk Processing: Upload a JSON or CSV file containing multiple customer records.

   - Template Support: Includes a "Download Sample CSV Template" button to ensure your input format is correct.

   - Analytics:

     - Visualizes churn distribution (Pie Chart).

     - Shows risk probability distribution (Histogram).

     - Provides a downloadable CSV of results with probability scores appended.

### Backend Implementation Details (Reference)

- **Pure Python Inference:** The Lambda function (refer to `part3_telco_churn_lambda_function.py`) manually implements:
  - Scaling: $x_{scaled} = \frac{x - mean}{scale}$
  - One-Hot Encoding: Dictionary lookups against the saved artifact.
  - Sigmoid Activation: $P(y=1) = \frac{1}{1 + e^{-z}}$
- **Warm Start Caching:** The function caches the model from S3 into memory (`global model_cache`) to speed up subsequent requests.









