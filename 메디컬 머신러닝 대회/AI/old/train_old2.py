import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score, mean_squared_error
import lightgbm as lgb
import joblib
import warnings
warnings.filterwarnings('ignore')

print("1. 기본 데이터 로드 중...")
ops  = pd.read_csv('operations.csv')
diag = pd.read_csv('diagnosis.csv')

# 진단명 피처 (환자별 진단 수 + 주요 질환군 분류)
diag_count = diag.groupby('subject_id')['icd10_cm'].count().reset_index()
diag_count.columns = ['subject_id', 'diag_count']

diag['icd_group'] = diag['icd10_cm'].apply(lambda c: str(c)[0].upper() if pd.notna(c) else '')
major_groups = {
    'diag_circulatory':    ['I'],
    'diag_neoplasm':       ['C','D'],
    'diag_endocrine':      ['E'],
    'diag_respiratory':    ['J'],
    'diag_digestive':      ['K'],
    'diag_musculoskeletal':['M'],
    'diag_genitourinary':  ['N'],
    'diag_injury':         ['S','T'],
}
for col, groups in major_groups.items():
    flag = diag[diag['icd_group'].isin(groups)].groupby('subject_id').size().reset_index()
    flag.columns = ['subject_id', col]
    flag[col] = (flag[col] > 0).astype(int)
    diag_count = diag_count.merge(flag, on='subject_id', how='left')
    diag_count[col] = diag_count[col].fillna(0).astype(int)

df = ops.merge(diag_count, on='subject_id', how='left')
df['diag_count'] = df['diag_count'].fillna(0)
for col in major_groups.keys():
    df[col] = df[col].fillna(0).astype(int)

encoders = {}
cat_cols = ['sex', 'race', 'department', 'antype', 'icd10_pcs']
for col in cat_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    encoders[col] = le

op_start = dict(zip(ops['subject_id'], ops['opstart_time']))

print("2. labs.csv 집계 중... (혈액검사 수치)")
lab_chunks = []
for chunk in pd.read_csv('labs.csv', chunksize=500000):
    chunk['op_start'] = chunk['subject_id'].map(op_start)
    chunk = chunk[chunk['chart_time'] < chunk['op_start']]
    lab_chunks.append(chunk)

labs = pd.concat(lab_chunks)
lab_pivot = labs.pivot_table(
    index='subject_id', columns='item_name',
    values='value', aggfunc=['mean', 'min', 'max']
)
lab_pivot.columns = ['lab_' + '_'.join(col) for col in lab_pivot.columns]
lab_pivot = lab_pivot.reset_index()
df = df.merge(lab_pivot, on='subject_id', how='left')
print(f"   혈액검사 피처: {len([c for c in lab_pivot.columns if c != 'subject_id'])}개")

print("3. ward_vitals 집계 중... (생체징후)")
ward_chunks = []
for chunk in pd.read_csv('ward_vitals.csv', chunksize=500000):
    chunk['op_start'] = chunk['subject_id'].map(op_start)
    chunk = chunk[chunk['chart_time'] < chunk['op_start']]
    ward_chunks.append(chunk)

ward = pd.concat(ward_chunks)
ward_pivot = ward.pivot_table(
    index='subject_id', columns='item_name',
    values='value', aggfunc=['mean', 'min', 'max', 'std']
)
ward_pivot.columns = ['_'.join(col) for col in ward_pivot.columns]
ward_pivot = ward_pivot.reset_index()
df = df.merge(ward_pivot, on='subject_id', how='left')

base_features    = ['age','sex','weight','height','asa','emop','department','antype','icd10_pcs','diag_count']
disease_features = list(major_groups.keys())
lab_cols         = [c for c in lab_pivot.columns  if c != 'subject_id']
vital_cols       = [c for c in ward_pivot.columns if c != 'subject_id']
features         = base_features + disease_features + lab_cols + vital_cols

df[features] = df[features].fillna(df[features].median())
medians = df[features].median().to_dict()
print(f"   총 피처: 기본 {len(base_features)} + 질환군 {len(disease_features)} + 혈액검사 {len(lab_cols)} + 생체징후 {len(vital_cols)} = {len(features)}개")

print("\n4. [고위험 환자 분류] 학습 중...")
df['icu_admission'] = df['icuin_time'].notna().astype(int)
X, y = df[features], df['icu_admission']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model_cls = lgb.LGBMClassifier(n_estimators=1000, learning_rate=0.03, num_leaves=63,
                                min_child_samples=30, class_weight='balanced', random_state=42)
model_cls.fit(X_train, y_train, eval_set=[(X_test, y_test)],
              callbacks=[lgb.early_stopping(50, verbose=False)])
auroc = roc_auc_score(y_test, model_cls.predict_proba(X_test)[:,1])
print(f"   AUROC: {auroc:.4f}")

print("\n5. [수술 시간 예측] 학습 중...")
df['surgery_duration'] = df['opend_time'] - df['opstart_time']
df_reg = df[df['surgery_duration'] > 0].copy()
X2, y2 = df_reg[features], df_reg['surgery_duration']
X_train2, X_test2, y_train2, y_test2 = train_test_split(X2, y2, test_size=0.2, random_state=42)
model_reg = lgb.LGBMRegressor(n_estimators=1000, learning_rate=0.03, num_leaves=63,
                               min_child_samples=30, random_state=42)
model_reg.fit(X_train2, y_train2, eval_set=[(X_test2, y_test2)],
              callbacks=[lgb.early_stopping(50, verbose=False)])
rmse = np.sqrt(mean_squared_error(y_test2, model_reg.predict(X_test2)))
print(f"   RMSE: {rmse:.2f}분")

print("\n6. 저장 중...")
joblib.dump(model_cls, 'model_cls.pkl')
joblib.dump(model_reg, 'model_reg.pkl')
joblib.dump(encoders,  'encoders.pkl')
joblib.dump(medians,   'medians.pkl')
joblib.dump(features,  'features.pkl')

print(f"\n 완료! AUROC {auroc:.4f} | RMSE {rmse:.2f}분")
print("포스터 반영: 진단명 + 혈액검사 수치 + 생체징후")
