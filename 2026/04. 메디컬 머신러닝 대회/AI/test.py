import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score, mean_squared_error
import lightgbm as lgb
import shap
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

print("1. 데이터 로드 중...")
ops = pd.read_csv('operations.csv')
diag = pd.read_csv('diagnosis.csv')

# 진단코드 피처
diag_count = diag.groupby('subject_id')['icd10_cm'].count().reset_index()
diag_count.columns = ['subject_id', 'diag_count']
df = ops.merge(diag_count, on='subject_id', how='left')
df['diag_count'] = df['diag_count'].fillna(0)

# 범주형 인코딩
cat_cols = ['sex', 'race', 'department', 'antype', 'icd10_pcs']
for col in cat_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))

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

# 피처 정의
base_features = ['age', 'sex', 'weight', 'height', 'asa', 'emop',
                 'department', 'antype', 'icd10_pcs', 'diag_count']
vital_cols = [c for c in ward_pivot.columns if c != 'subject_id']
features = base_features + vital_cols
df[features] = df[features].fillna(df[features].median())
print(f"총 피처 수: {len(features)}")

# ── 3번: 고위험 환자 분류 ──────────────────
print("\n3. [3번] 고위험 환자 분류 학습 중...")
df['icu_admission'] = df['icuin_time'].notna().astype(int)
X = df[features]
y = df['icu_admission']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model_cls = lgb.LGBMClassifier(n_estimators=500, learning_rate=0.05,
                                num_leaves=63, random_state=42)
model_cls.fit(X_train, y_train)
auroc = roc_auc_score(y_test, model_cls.predict_proba(X_test)[:, 1])
print(f"✅ [3번] AUROC: {auroc:.4f}")

# ── 4번: 수술 시간 예측 ────────────────────
print("\n4. [4번] 수술 시간 예측 학습 중...")
df['surgery_duration'] = df['opend_time'] - df['opstart_time']
df_reg = df[df['surgery_duration'] > 0].copy()
X2 = df_reg[features]
y2 = df_reg['surgery_duration']
X_train2, X_test2, y_train2, y_test2 = train_test_split(X2, y2, test_size=0.2, random_state=42)

model_reg = lgb.LGBMRegressor(n_estimators=500, learning_rate=0.05,
                               num_leaves=63, random_state=42)
model_reg.fit(X_train2, y_train2)
rmse = np.sqrt(mean_squared_error(y_test2, model_reg.predict(X_test2)))
print(f"✅ [4번] RMSE: {rmse:.2f} 분")

# ── SHAP 시각화 ───────────────────────────
print("\n5. SHAP 분석 중...")

# 3번 SHAP
explainer_cls = shap.TreeExplainer(model_cls)
shap_values_cls = explainer_cls.shap_values(X_test.iloc[:500])

plt.figure()
shap.summary_plot(
    shap_values_cls[:, :, 1] if len(np.array(shap_values_cls).shape) == 3 else shap_values_cls,
    X_test.iloc[:500], show=False
)
plt.title("고위험 환자 분류 - 변수 중요도")
plt.tight_layout()
plt.savefig('shap_classification.png', dpi=150)
plt.close()

# 4번 SHAP
explainer_reg = shap.TreeExplainer(model_reg)
shap_values_reg = explainer_reg.shap_values(X_test2.iloc[:500])

plt.figure()
shap.summary_plot(shap_values_reg, X_test2.iloc[:500], show=False)
plt.title("수술 시간 예측 - 변수 중요도")
plt.tight_layout()
plt.savefig('shap_regression.png', dpi=150)
plt.close()

print("✅ SHAP 이미지 저장 완료! (shap_classification.png, shap_regression.png)")
print("\n🎉 전체 완료!")