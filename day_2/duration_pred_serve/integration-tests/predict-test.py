import requests

# url = "http://127.0.0.1:9696/predict"
url = "https://duration-pred-serve-2025-04.fly.dev/predict"
trip = {
	"PULocationID": 43,
	"DOLocationID": 238,
	"trip_distance": 1.16
}
response = requests.post(url, json=trip).json()
print(response)