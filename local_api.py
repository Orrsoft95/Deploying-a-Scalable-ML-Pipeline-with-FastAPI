import json

import requests

# DONE: send a GET using the URL http://127.0.0.1:8000
r = requests.get(url="http://127.0.0.1:8000")

# DONE: print the status code
print(f"Status Code: {r.status_code}")
# DONE: print the welcome message
response = r.json()
response = response["greeting"]
print(f"Result: {response}")



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

# DONE: send a POST using the data above
r = requests.post(
    url="http://127.0.0.1:8000/data/",
    data=json.dumps(data)
    )

# DONE: print the status code
print(f"Statsu Code: {r.status_code}")
# DONE: print the result
result = r.json()
result = result["result"]
print(f"Result: {result}")
