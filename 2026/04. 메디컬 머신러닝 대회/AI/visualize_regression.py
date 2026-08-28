import pandas as pd
import matplotlib.pyplot as plt
import joblib
import platform
import shap
import warnings
warnings.filterwarnings('ignore')

print("=" * 55)
print("  특정 환자 수술 시간 예측 원인 분석 (SHAP XAI)")
print("=" * 55)

# 1. 한글 폰트 및 마이너스 기호 깨짐 방지 설정
system_os = platform.system()
if system_os == 'Windows':
    plt.rc('font', family='Malgun Gothic')
elif system_os == 'Darwin':
    plt.rc('font', family='AppleGothic')
else:
    plt.rc('font', family='NanumGothic')
plt.rc('axes', unicode_minus=False)

# 2. 모델 및 피처, 중간값 데이터 로드
try:
    model_reg = joblib.load('model_reg.pkl')
    features  = joblib.load('features.pkl')
    medians   = joblib.load('medians.pkl')
    print("✅ 모델 및 데이터 로딩 완료!")
except Exception as e:
    print(f"⚠ 파일 로드 실패: {e}")
    exit()

# 3. 분석할 가상의 특정 환자 데이터 만들기 (Web UI에서 넘어온 데이터라 가정)
# 일단 모든 값을 전체 환자의 중앙값(median)으로 채웁니다.
patient_data = {f: medians.get(f, 0) for f in features}

# 특징적인 값을 몇 개 덮어씌웁니다 (예: 75세 고령, ASA 4등급의 중증 환자)
patient_data.update({
    'age': 75,             # 고령
    'asa': 4,              # 높은 수술 위험도
    'diag_count': 6,       # 기저질환 많음
    'weight': 85,          # 체중
    'height': 165,
    'mean_hr': 95          # 심박수 약간 높음
})

# 데이터프레임으로 변환 (반드시 학습할 때와 동일한 features 순서 유지)
X_sample = pd.DataFrame([patient_data])[features]

# 4. SHAP TreeExplainer로 모델 분석
print("🔍 SHAP 값을 계산 중입니다...")
explainer = shap.TreeExplainer(model_reg)
shap_values = explainer(X_sample)

# 5. 차트에 출력될 피처명 한글화
FEATURE_KR = {
    'icd10_pcs': '수술 종류', 'department': '진료과', 'age': '환자 나이',
    'asa': 'ASA 등급', 'weight': '체중', 'height': '신장',
    'antype': '마취 유형', 'emop': '응급 수술', 'diag_count': '기저질환 수',
    'surgery_rank': '수술 순번', 'mean_hr': '평균 심박수',
    'std_hr': '심박수 변동성', 'mean_bt': '평균 체온', 'mean_rr': '평균 호흡수'
}
# 영어로 된 피처명을 한글로 매핑 (없는 건 원래 이름 유지)
kr_feature_names = [FEATURE_KR.get(f, f[:15]) for f in features]
shap_values.feature_names = kr_feature_names

# 6. 폭포수(Waterfall) 차트 그리기
plt.figure(figsize=(12, 8))

# max_display=10 은 가장 영향력이 컸던 상위 10개 변수만 보여준다는 뜻입니다.
shap.plots.waterfall(shap_values[0], max_display=10, show=False)

plt.title('해당 환자의 수술 시간 예측 인자 상세 분석 (XAI)', fontsize=16, fontweight='bold', pad=20)
plt.tight_layout()

# 7. 차트 저장
out_file = 'patient_shap_waterfall.png'
plt.savefig(out_file, dpi=200, bbox_inches='tight')
plt.close()
print(f"✅ 분석 완료! '{out_file}' 파일을 확인해주세요.")