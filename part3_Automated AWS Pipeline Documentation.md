## Architecture Diagram 

```

```





## Steps to Reproduce

**1. Cloud Storage Setup:**

- Created a private Amazon S3 bucket named `dsci352-telco-churn-project` in the `us-east-1` region.
- This bucket serves as the model registry, hosting the serialized model artifacts.

**2. AWS IAM Setup**

- Created a programmatic IAM user (`telco-script-user`) with `AmazonS3FullAccess`.
- Generated Access Key ID & Secret Access Key and configured them locally using `boto3`. This authorized the local Python training script to upload files securely to the private bucket.

**3. Model Training & Export:**

- Executed the local training script (`finalproject_DSCI-352_Fall2025_Starter.py`) with `ENABLE_AWS_EXPORT = True`.
- The script trained a  SGDClassifier with log_loss and extracted the raw coefficients (weights) and preprocessing statistics (means, scales) into a lightweight JSON file (`telco_churn_light.json`).
- Used the `boto3` library to automatically upload this JSON artifact to the S3 bucket.

**4. Lambda Function Deployment**

- Created an AWS Lambda function named `DSCI352_PredictTelcoChurn` using the Python 3.10 runtime and `x86_64` architecture.
- Implemented a custom Pure Python inference script. To avoid the cold-start latency and size limits of installing libraries like `pandas` or `scikit-learn` in Lambda, we manually implemented:
  - One-Hot Encoding lookup.
  - Standard Scaling math.
  - The Logistic Sigmoid function: $P(y=1) = \frac{1}{1 + e^{-z}}$.

**5. Access Configuration**

- Modified the Lambda Execution Role in AWS IAM to include the `AmazonS3ReadOnlyAccess` policy. This granted the function permission to read the model file from the private S3 bucket.
- This granted the Lambda function strict permission to read the model file from the private S3 bucket without allowing public access.

**6. API Exposure**

- Configured an Amazon API Gateway (HTTP API) trigger for the Lambda function.
- API endpoint: https://yhj9s420fi.execute-api.us-east-1.amazonaws.com/default/DSCI352_PredictTelcoChurn
- This turns the Python function into a web-accessible REST API that accepts JSON payloads.

**7. Streamlit Dashboard**

- Developed an interactive web application (`part3_telco_churn_dashboard.py`) using Streamlit.
- Features
  - Single Customer: A form interface to adjust customer parameters (Tenure, Contract, etc.) and get real-time risk scores.
  - Batch Prediction: Capability to upload a CSV (e.g., `telco_test_sample.csv`), process it against the API, and download results.
- Execution: `streamlit run part3_telco_churn_dashboard.py`

