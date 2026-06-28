import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import platform

# ==========================================
# 1. 시각화할 성능 지표 데이터 설정
# ==========================================
# 'train.py' 실행 결과로 나온 실제 수치로 수정해주세요.
metrics_data = {
    '지표': ['AUROC', 'Accuracy', 'Precision', 'Recall', 'F1-Score'],
    '수치': [0.9745, 0.8850, 0.8520, 0.9130, 0.8815] # 실제 값으로 변경하세요.
}

# 데이터를 pandas DataFrame으로 변환합니다.
df_metrics = pd.DataFrame(metrics_data)

# ==========================================
# 2. 그래프 스타일 및 한글 폰트 설정
# ==========================================
# 배경 스타일 설정
sns.set_theme(style="whitegrid") 

# 운영체제에 맞는 한글 폰트 강제 적용 (Seaborn 설정 후 적용해야 안 깨집니다)
system_os = platform.system()
if system_os == 'Windows':
    plt.rc('font', family='Malgun Gothic')  # 윈도우: 맑은 고딕
elif system_os == 'Darwin':
    plt.rc('font', family='AppleGothic')    # 맥: 애플 고딕
else:
    plt.rc('font', family='NanumGothic')    # 리눅스: 나눔 고딕

plt.rc('axes', unicode_minus=False)         # 마이너스 기호 깨짐 방지

# ==========================================
# 3. 그래프 그리기
# ==========================================
plt.figure(figsize=(10, 6))      # 그래프 크기 설정

# Seaborn 막대그래프 생성
barplot = sns.barplot(x='지표', y='수치', data=df_metrics, color='skyblue')

# 그래프 제목과 축 레이블 설정
plt.title('고위험 환자 분류 모델 성능 지표', fontsize=16, fontweight='bold', pad=15)
plt.ylabel('수치', fontsize=12)
plt.xlabel('성능 지표', fontsize=12)

# y축 범위를 0 ~ 1.1 로 설정 (막대 위의 글자가 잘리지 않도록 여유 공간 확보)
plt.ylim(0, 1.1)

# ==========================================
# 4. 막대 위에 실제 수치 표시
# ==========================================
for p in barplot.patches:
    barplot.annotate(format(p.get_height(), '.4f'), # 소수점 4자리까지 표시
                     (p.get_x() + p.get_width() / 2., p.get_height()), # 표시 위치 (막대 중앙 위)
                     ha = 'center', va = 'center', 
                     xytext = (0, 9), # 막대와 텍스트 사이의 간격
                     textcoords = 'offset points',
                     fontsize=11, fontweight='bold')

# 여백을 깔끔하게 자동 조정
plt.tight_layout()

# 그래프 출력
plt.show()