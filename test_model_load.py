import os
import sys

print(f"CWD: {os.getcwd()}")
print(f"models/churn_model.pkl exists: {os.path.exists('models/churn_model.pkl')}")

sys.path.insert(0, 'src')

print("Trying to import app.main...")
from app.main import app, predictor, model_loaded

print(f"After import - Model loaded: {model_loaded}")
print(f"After import - Predictor: {predictor}")

print("\nCreating TestClient...")
from fastapi.testclient import TestClient
client = TestClient(app)

print(f"After TestClient - Model loaded: {model_loaded}")
print(f"After TestClient - Predictor: {predictor}")

