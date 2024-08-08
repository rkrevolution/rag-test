import requests

url = "http://localhost:5001/query"
data = {"input": "What is the theme of the book?"}

response = requests.post(url, json=data)

print(f"Status Code: {response.status_code}")
print(f"Response Content: {response.text}")

try:
    json_response = response.json()
    print("JSON Response:")
    print(json_response)
    if 'error' in json_response:
        print(f"Error: {json_response['error']}")
    elif 'results' in json_response:
        print("Top 5 relevant chunks:")
        for i, chunk in enumerate(json_response['results'], 1):
            print(f"{i}. {chunk[:100]}...")  # Print first 100 characters of each chunk
    else:
        print("Unexpected response format")
except requests.exceptions.JSONDecodeError:
    print("Failed to decode JSON. Raw response:")
    print(response.text)
