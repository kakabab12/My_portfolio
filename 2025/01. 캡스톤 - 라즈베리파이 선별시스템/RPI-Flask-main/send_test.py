import requests
url = "http://192.168.0.8:9999/test"

payload = {
        "class": "test_box", 
        "conf": 0.98,
        "x": 500,
        "y": 610
}

try:
    res = requests.post(url,json=payload)
    print("rqquest code:", res.status_code )    
    print("request:", res.text)
except Exception as e:
    print("send fail:", e)
    