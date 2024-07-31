import requests

url = "http://localhost:8000/predict"
data = {
    "text": "Fuck you bitch "
}

response = requests.post(url, json=data)
print(response.json())
