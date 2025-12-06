import json
import boto3
import math
import os

# --- CONFIGURATION ---
# Connect to S3
BUCKET_NAME = "dsci352-telco-churn-project" 
MODEL_KEY = "models/telco_churn_light.json"

s3 = boto3.client('s3')

# Global variable to cache the model in memory (Warm Start)
model_cache = None

def load_model():
    """Downloads model from S3 if not already loaded."""
    global model_cache
    if model_cache:
        return model_cache
        
    print(f"Downloading model from s3://{BUCKET_NAME}/{MODEL_KEY}...")
    try:
        response = s3.get_object(Bucket=BUCKET_NAME, Key=MODEL_KEY)
        content = response['Body'].read().decode('utf-8')
        data = json.loads(content)
        # The starter script wraps the artifact in a key called "artifact"
        model_cache = data["artifact"]
        return model_cache
    except Exception as e:
        print(f"Error downloading model: {e}")
        raise e

def sigmoid(z):
    if z < -709: return 0.0
    if z > 709: return 1.0
    return 1.0 / (1.0 + math.exp(-z))

def predict(data, model):
    # 1. Preprocess Numerics
    # Map input keys to the order the model expects
    num_vec = []
    for i, name in enumerate(model["numeric_features"]):
        val = data.get(name)
        # Impute missing with saved median
        if val is None: val = model["numeric_imputer_medians"][i]
        else: val = float(val)
        # Scale: (val - mean) / scale
        val = (val - model["numeric_means"][i]) / model["numeric_scales"][i]
        num_vec.append(val)

    # 2. Preprocess Categoricals (Manual One-Hot)
    cat_vec = []
    for i, name in enumerate(model["categorical_features"]):
        val = str(data.get(name, "Missing"))
        categories = model["categories"][i]
        
        # Create zero vector
        ohe = [0.0] * len(categories)
        if val in categories:
            idx = categories.index(val)
            ohe[idx] = 1.0
        cat_vec.extend(ohe)

    # 3. Dot Product
    # z = intercept + (w1*x1) + (w2*x2) ...
    full_vec = num_vec + cat_vec
    coefs = model["coef"] 
    intercept = model["intercept"][0]
    
    z = intercept
    for x, w in zip(full_vec, coefs):
        z += x * w
        
    prob = sigmoid(z)
    return prob

def lambda_handler(event, context):
    print("Force Reloading Model...")
    try:
        # 1. Load Model
        model = load_model()
        
        # 2. Parse Input
        # API Gateway sends input in 'body' as a string
        if 'body' in event:
            body = event['body']
            if isinstance(body, str):
                input_data = json.loads(body)
            else:
                input_data = body
        else:
            # Direct invocation (Test tab)
            input_data = event

        # 3. Predict
        prob = predict(input_data, model)
        prediction = "Yes" if prob > 0.5 else "No"
        
        # 4. Return Result
        return {
            'statusCode': 200,
            'headers': {'Content-Type': 'application/json'},
            'body': json.dumps({
                "churn_probability": round(prob, 4),
                "prediction": prediction
            })
        }
        
    except Exception as e:
        return {
            'statusCode': 500,
            'body': json.dumps({"error": str(e)})
        }