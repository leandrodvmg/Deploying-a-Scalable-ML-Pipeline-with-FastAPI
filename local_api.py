import json
import requests

# URL for the local FastAPI server
BASE_URL = "http://127.0.0.1:8000"

# Send a GET request to the root endpoint and print status and welcome message
try:
    r = requests.get(BASE_URL)
    print("GET status:", r.status_code)
    try:
        payload = r.json()
        # welcome message commonly under 'message' or similar; print full payload if unsure
        print("GET response:", json.dumps(payload, indent=2))
    except ValueError:
        print("GET response (text):", r.text)
except requests.exceptions.RequestException as e:
    print("GET request failed:", e)


data = {
    "age": 37,
    "workclass": "Private",
    "fnlgt": 178356,
    "education": "HS-grad",
    "education-num": 10,
    "marital-status": "Married-civ-spouse",
    "occupation": "Prof-specialty",
    "relationship": "Husband",
    "race": "White",
    "sex": "Male",
    "capital-gain": 0,
    "capital-loss": 0,
    "hours-per-week": 40,
    "native-country": "United-States",
}

# Send a POST request to the prediction endpoint and print status and result
predict_url = f"{BASE_URL}/data/"
try:
    r = requests.post(predict_url, json=data)
    print("POST status:", r.status_code)
    try:
        payload = r.json()
        print("POST response:", json.dumps(payload, indent=2))
    except ValueError:
        print("POST response (text):", r.text)
except requests.exceptions.RequestException as e:
    print("POST request failed:", e)
