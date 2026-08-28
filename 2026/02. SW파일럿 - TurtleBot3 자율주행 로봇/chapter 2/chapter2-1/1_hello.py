import os
import sys

def create_hello_file():
    target_dir = "/test"
    file_path = os.path.join(target_dir, "hello.txt")
    
    print(f"[시스템] {file_path} 생성을 시작합니다.")
    
    # [수행단계 핵심 - 예외 처리] 
    # 관리자 권한이 없으면 프로그램이 튕기지 않고 안내 메시지를 출력하도록 예외 처리 적용
    try:
        # 폴더가 없으면 생성
        if not os.path.exists(target_dir):
            os.makedirs(target_dir, exist_ok=True)
            
        # 파일 작성
        with open(file_path, "w", encoding="utf-8") as f:
            f.write("Hello Linux")
            
        print("[성공] 파일이 예외 없이 성공적으로 생성되었습니다!")
        print(f"[확인 방법] 터미널에 'cat {file_path}'를 입력하여 내용을 확인하세요.")
        
    except PermissionError:
        print("[예외 발생] 권한이 없습니다! (Permission Denied)")
        print("[해결 방법] 리눅스 루트(/) 디렉토리는 관리자 권한이 필요합니다.")
        print("            'sudo python3 1_hello.py' 명령어로 다시 실행해 주세요.")
    except Exception as e:
        print(f"[기타 예외 발생] {e}")

if __name__ == "__main__":
    create_hello_file()