import os
import google.genai as genai
import sys
import io

# 윈도우 터미널 인코딩 강제 설정 (UTF-8)
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# 본인의 API 키를 입력하세요
# API 키는 환경변수로 받는다 -- 코드에 키를 직접 적으면 저장소를 공개하는
# 순간 그대로 노출된다. 실행 전에 GEMINI_API_KEY 를 설정할 것.
#   (Windows) set GEMINI_API_KEY=발급받은_키
#   (Linux)   export GEMINI_API_KEY=발급받은_키
client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY", ""))

print("=== Gemini Model List (English Only) ===")
try:
    # 모델 목록을 가져옵니다.
    model_list = client.models.list()
    
    for m in model_list:
        # 모델 이름에서 'models/' 부분을 제외한 순수 이름만 추출
        name = m.name.replace("models/", "")
        
        # ASCII가 아닌 문자가 포함될 경우를 대비해 필터링 후 출력
        safe_name = name.encode('ascii', 'ignore').decode('ascii')
        print(f"Name: {safe_name}")

except Exception as e:
    # 에러 발생 시 영문으로만 에러 메시지 출력
    print(f"Error: {str(e).encode('ascii', 'ignore').decode('ascii')}")