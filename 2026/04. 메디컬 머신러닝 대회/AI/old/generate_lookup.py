"""
train.py 전체 재실행 없이 수술 코드 목록만 빠르게 생성하는 스크립트
"""
import pandas as pd
import joblib
import warnings
warnings.filterwarnings('ignore')

print("수술 코드 목록 생성 중...")

encoders = joblib.load('encoders.pkl')
ops = pd.read_csv('operations.csv')

ops['surgery_duration'] = ops['opend_time'] - ops['opstart_time']
ops = ops[(ops['surgery_duration'] >= 5) & (ops['surgery_duration'] <= 1440)]

dept_le = encoders['department']
pcs_le  = encoders['icd10_pcs']

ops['dept_code']  = dept_le.inverse_transform(ops['department'].apply(
    lambda x: min(int(x), len(dept_le.classes_)-1) if pd.notna(x) else 0
))
ops['pcs_code'] = pcs_le.inverse_transform(ops['icd10_pcs'].apply(
    lambda x: min(int(x), len(pcs_le.classes_)-1) if pd.notna(x) else 0
))

lookup = (
    ops.groupby(['dept_code', 'pcs_code'])
    .agg(count=('subject_id', 'count'), avg_duration=('surgery_duration', 'mean'))
    .reset_index()
    .sort_values('count', ascending=False)
)

# 진료과별 상위 30개
top = (
    lookup.groupby('dept_code')
    .apply(lambda x: x.nlargest(30, 'count'))
    .reset_index(drop=True)
)
top.columns = ['department_name', 'icd10_pcs_code', 'count', 'avg_duration']
top['avg_duration'] = top['avg_duration'].round(1)

joblib.dump(top, 'icd10_pcs_lookup.pkl')
print(f"✅ 완료! {len(top)}개 저장")
print(top.head(10).to_string(index=False))
