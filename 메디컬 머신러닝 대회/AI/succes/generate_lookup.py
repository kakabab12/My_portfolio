"""
train.py 전체 재실행 없이 수술 코드 목록만 빠르게 생성하는 스크립트
"""
import pandas as pd
import joblib
import warnings
warnings.filterwarnings('ignore')

print("수술 코드 목록 생성 중...")

ops = pd.read_csv('operations.csv')

ops['surgery_duration'] = ops['opend_time'] - ops['opstart_time']
ops = ops[(ops['surgery_duration'] >= 5) & (ops['surgery_duration'] <= 1440)]

# 에러가 나던 역변환(inverse_transform) 부분 삭제
# 원본 operations.csv 에는 이미 'department'와 'icd10_pcs'가 문자열로 들어있습니다!

lookup = (
    ops.groupby(['department', 'icd10_pcs']) # 원본 컬럼명 사용
    .agg(count=('subject_id', 'count'), avg_duration=('surgery_duration', 'mean'))
    .reset_index()
    .sort_values('count', ascending=False)
)

# 진료과별 상위 30개
top = (
    lookup.groupby('department') # 원본 컬럼명 사용
    .apply(lambda x: x.nlargest(30, 'count'))
    .reset_index(drop=True)
)

# 컬럼 이름을 프론트엔드/서버에서 원하는 형태로 변경
top.columns = ['department_name', 'icd10_pcs_code', 'count', 'avg_duration']
top['avg_duration'] = top['avg_duration'].round(1)

joblib.dump(top, 'icd10_pcs_lookup.pkl')
print(f"✅ 완료! {len(top)}개 저장")
print(top.head(10).to_string(index=False))