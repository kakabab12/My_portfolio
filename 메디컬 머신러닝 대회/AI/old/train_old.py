import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score, mean_squared_error
import lightgbm as lgb
import joblib
import warnings
warnings.filterwarnings('ignore')

print("1. 데이터 로드 중...")
ops = pd.read_csv('operations.csv')
diag = pd.read_csv('diagnosis.csv')

diag_count = diag.groupby('subject_id')['icd10_cm'].count().reset_index()
diag_count.columns = ['subject_id', 'diag_count']
df = ops.merge(diag_count, on='subject_id', how='left')
df['diag_count'] = df['diag_count'].fillna(0)

print("2. ward_vitals 집계 중... (시간 걸림)")
op_start = dict(zip(ops['subject_id'], ops['opstart_time']))
chunk_list = []
for chunk in pd.read_csv('ward_vitals.csv', chunksize=500000):
    chunk['op_start'] = chunk['subject_id'].map(op_start)
    chunk = chunk[chunk['chart_time'] < chunk['op_start']]
    chunk_list.append(chunk)

ward = pd.concat(chunk_list)
ward_pivot = ward.pivot_table(
    index='subject_id', columns='item_name',
    values='value', aggfunc=['mean', 'min', 'max', 'std']
)
ward_pivot.columns = ['_'.join(col) for col in ward_pivot.columns]
ward_pivot = ward_pivot.reset_index()
df = df.merge(ward_pivot, on='subject_id', how='left')

# 인코더 학습 & 저장
encoders = {}
cat_cols = ['sex', 'race', 'department', 'antype', 'icd10_pcs']
for col in cat_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    encoders[col] = le

base_features = ['age', 'sex', 'weight', 'height', 'asa', 'emop',
                 'department', 'antype', 'icd10_pcs', 'diag_count']
vital_cols = [c for c in ward_pivot.columns if c != 'subject_id']
features = base_features + vital_cols
df[features] = df[features].fillna(df[features].median())

# 결측치 대체용 중앙값 저장
medians = df[features].median().to_dict()

print("3. 고위험 환자 분류 모델 학습 중...")
df['icu_admission'] = df['icuin_time'].notna().astype(int)
X, y = df[features], df['icu_admission']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model_cls = lgb.LGBMClassifier(n_estimators=500, learning_rate=0.05, num_leaves=63, random_state=42)
model_cls.fit(X_train, y_train)
auroc = roc_auc_score(y_test, model_cls.predict_proba(X_test)[:, 1])
print(f"✅ [분류] AUROC: {auroc:.4f}")

print("4. 수술 시간 예측 모델 학습 중...")
df['surgery_duration'] = df['opend_time'] - df['opstart_time']
df_reg = df[df['surgery_duration'] > 0].copy()
X2, y2 = df_reg[features], df_reg['surgery_duration']
X_train2, X_test2, y_train2, y_test2 = train_test_split(X2, y2, test_size=0.2, random_state=42)
model_reg = lgb.LGBMRegressor(n_estimators=500, learning_rate=0.05, num_leaves=63, random_state=42)
model_reg.fit(X_train2, y_train2)
rmse = np.sqrt(mean_squared_error(y_test2, model_reg.predict(X_test2)))
print(f"✅ [회귀] RMSE: {rmse:.2f}분")

print("5. 모델 저장 중...")
joblib.dump(model_cls, 'model_cls.pkl')
joblib.dump(model_reg, 'model_reg.pkl')
joblib.dump(encoders,  'encoders.pkl')
joblib.dump(medians,   'medians.pkl')
joblib.dump(features,  'features.pkl')

print("🎉 완료! pkl 파일 5개 생성됨")
