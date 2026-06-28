import requests

response = requests.get("http://localhost:5000/data")
print("응답 상태:", response.status_code)
print("응답 내용:", response.text)

# JSON 변환은 이후에
try:
    data = response.json()
    print("JSON 데이터:", data)
except Exception as e:
    print("JSON 파싱 실패:", e)
