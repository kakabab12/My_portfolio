import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, KFold, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score, mean_squared_error
import lightgbm as lgb
import joblib
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("  Medical AI Dataton 2026 - 모델 학습")
print("=" * 60)

# ── 1. Operations 로드 & 기본 정제 ──────────────────────────
print("\n[1/7] operations.csv 로드 중...")
ops = pd.read_csv('operations.csv')
print(f"  원본: {len(ops):,}건")

# 수술 시간 계산 (분)
ops['surgery_duration'] = ops['opend_time'] - ops['opstart_time']

# 이상 수술 시간 제거 (5분 미만, 1440분=24시간 초과)
ops = ops[(ops['surgery_duration'] >= 5) & (ops['surgery_duration'] <= 1440)]
print(f"  수술 시간 필터(5~1440분) 후: {len(ops):,}건")

# ── 핵심: 환자별 첫 번째 수술만 기준점으로 사용 (미래 데이터 차단) ──
# 예측 시점 = 수술 시작 직전이므로 각 수술의 opstart_time 사용
# 단, 이전 수술 정보가 현재 수술 예측에 새어 들어가지 않도록
# 각 수술에 대해 "해당 수술 시작 전" 데이터만 사용
ops = ops.sort_values(['subject_id', 'opstart_time'])
ops['surgery_rank'] = ops.groupby('subject_id').cumcount() + 1  # 몇 번째 수술인지

print(f"  첫 수술 비율: {(ops['surgery_rank']==1).mean()*100:.1f}%")

# 각 수술의 cutoff = 해당 수술의 opstart_time
ops['cutoff_time'] = ops['opstart_time']

# ── 2. Diagnosis 로드 & 진단명 피처 ────────────────────────
print("\n[2/7] diagnosis.csv 로드 중...")
diag = pd.read_csv('diagnosis.csv')
print(f"  진단 레코드: {len(diag):,}건")

# 진단 컬럼 확인
print(f"  컬럼: {list(diag.columns)}")

# 환자별 진단 수 (수술과 무관하게 전체 - 입원 진단이므로 수술 전으로 간주)
diag_count = diag.groupby('subject_id')['icd10_cm'].count().reset_index()
diag_count.columns = ['subject_id', 'diag_count']

# 주요 질환군 (ICD-10 첫 글자 기반)
diag['icd_group'] = diag['icd10_cm'].apply(
    lambda c: str(c)[0].upper() if pd.notna(c) and len(str(c)) > 0 else 'Z'
)

disease_map = {
    'diag_circulatory':     ['I'],
    'diag_neoplasm':        ['C', 'D'],
    'diag_endocrine':       ['E'],
    'diag_respiratory':     ['J'],
    'diag_digestive':       ['K'],
    'diag_musculoskeletal': ['M'],
    'diag_genitourinary':   ['N'],
    'diag_injury':          ['S', 'T'],
    'diag_nervous':         ['G'],
    'diag_mental':          ['F'],
}

for col, groups in disease_map.items():
    flag = diag[diag['icd_group'].isin(groups)].groupby('subject_id').size()
    diag_count[col] = diag_count['subject_id'].map(flag).fillna(0).clip(upper=1).astype(int)

# ops와 merge (subject_id 기준)
df = ops.merge(diag_count, on='subject_id', how='left')
for col in ['diag_count'] + list(disease_map.keys()):
    df[col] = df[col].fillna(0)

# ── 3. 범주형 인코딩 ────────────────────────────────────────
print("\n[3/7] 범주형 인코딩 중...")
encoders = {}
cat_cols = ['sex', 'race', 'department', 'antype', 'icd10_pcs']
for col in cat_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    encoders[col] = le
    print(f"  {col}: {len(le.classes_)}개 클래스")

# ── 4. Labs (혈액검사) ──────────────────────────────────────
print("\n[4/7] labs.csv 집계 중... (수술 전 혈액검사 수치)")
op_cutoff = dict(zip(ops['opid'], ops['cutoff_time'])) if 'opid' in ops.columns else None
op_start_by_subject = dict(zip(ops['subject_id'], ops['opstart_time']))

lab_chunks = []
for chunk in pd.read_csv('labs.csv', chunksize=500000):
    chunk['cutoff'] = chunk['subject_id'].map(op_start_by_subject)
    chunk = chunk[chunk['chart_time'] < chunk['cutoff']]
    lab_chunks.append(chunk)

labs = pd.concat(lab_chunks, ignore_index=True)
print(f"  수술 전 검사 레코드: {len(labs):,}건")
print(f"  검사 항목: {labs['item_name'].nunique()}개")
print(f"  주요 항목: {labs['item_name'].value_counts().head(10).index.tolist()}")

# 검사항목별 mean/min/max (최근 값이 가장 중요하므로 last도 추가)
lab_pivot = labs.pivot_table(
    index='subject_id', columns='item_name',
    values='value', aggfunc=['mean', 'min', 'max', 'last']
)
lab_pivot.columns = ['lab_' + agg + '_' + col for agg, col in lab_pivot.columns]
lab_pivot = lab_pivot.reset_index()
df = df.merge(lab_pivot, on='subject_id', how='left')
lab_cols = [c for c in lab_pivot.columns if c != 'subject_id']
print(f"  혈액검사 피처: {len(lab_cols)}개")

# ── 5. Ward Vitals (병동 생체징후) ─────────────────────────
print("\n[5/7] ward_vitals.csv 집계 중...")
ward_chunks = []
for chunk in pd.read_csv('ward_vitals.csv', chunksize=500000):
    chunk['cutoff'] = chunk['subject_id'].map(op_start_by_subject)
    chunk = chunk[chunk['chart_time'] < chunk['cutoff']]
    ward_chunks.append(chunk)

ward = pd.concat(ward_chunks, ignore_index=True)
print(f"  수술 전 생체징후 레코드: {len(ward):,}건")

ward_pivot = ward.pivot_table(
    index='subject_id', columns='item_name',
    values='value', aggfunc=['mean', 'min', 'max', 'std', 'last']
)
ward_pivot.columns = ['_'.join([agg, col]) for agg, col in ward_pivot.columns]
ward_pivot = ward_pivot.reset_index()
df = df.merge(ward_pivot, on='subject_id', how='left')
vital_cols = [c for c in ward_pivot.columns if c != 'subject_id']
print(f"  생체징후 피처: {len(vital_cols)}개")

# ── 6. 피처 최종 구성 ──────────────────────────────────────
print("\n[6/7] 피처 구성 및 전처리 중...")

base_features = [
    'age', 'sex', 'weight', 'height', 'asa', 'emop',
    'department', 'antype', 'icd10_pcs',
    'surgery_rank',   # 몇 번째 수술인지
]
disease_features = list(disease_map.keys())
features = base_features + ['diag_count'] + disease_features + lab_cols + vital_cols

# 존재하는 컬럼만
features = [f for f in features if f in df.columns]

# 결측치: 중앙값으로
df[features] = df[features].fillna(df[features].median())

# 이상치 클리핑 (1-99 퍼센타일)
for col in lab_cols + vital_cols:
    if col in df.columns:
        lo = df[col].quantile(0.01)
        hi = df[col].quantile(0.99)
        df[col] = df[col].clip(lo, hi)

medians = df[features].median().to_dict()

print(f"  최종 피처 수: {len(features)}")
print(f"  - 기본: {len(base_features)}")
print(f"  - 진단명(질환군): {len(disease_features)}")
print(f"  - 혈액검사: {len(lab_cols)}")
print(f"  - 생체징후: {len(vital_cols)}")

# ── 7. 모델 학습 ────────────────────────────────────────────
print("\n[7/7] 모델 학습 중...")

X = df[features]

# ── 고위험 환자 분류 ──
print("\n  [분류] 고위험 환자 (ICU 입원) 예측...")
df['icu_admission'] = df['icuin_time'].notna().astype(int)
y_cls = df['icu_admission']
print(f"  ICU 입원: {y_cls.sum():,}명 ({y_cls.mean()*100:.1f}%)")

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y_cls, test_size=0.2, random_state=42, stratify=y_cls
)

model_cls = lgb.LGBMClassifier(
    n_estimators=2000,
    learning_rate=0.02,
    num_leaves=63,
    min_child_samples=50,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=0.1,
    class_weight='balanced',
    random_state=42,
    verbose=-1,
)
model_cls.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(period=-1)]
)
auroc = roc_auc_score(y_test, model_cls.predict_proba(X_test)[:, 1])
print(f"  ✅ AUROC: {auroc:.4f}")

# SHAP 중요도 상위 20개 출력
imp = pd.Series(model_cls.feature_importances_, index=features).sort_values(ascending=False)
print("  상위 10 중요 피처:")
for feat, val in imp.head(10).items():
    print(f"    {feat}: {val:.0f}")

# ── 수술 시간 예측 ──
print("\n  [회귀] 수술 시간 예측...")
y_reg = df['surgery_duration']
print(f"  수술 시간 분포: {y_reg.describe()[['min','mean','50%','max']].to_dict()}")

X_train2, X_test2, y_train2, y_test2 = train_test_split(
    X, y_reg, test_size=0.2, random_state=42
)

model_reg = lgb.LGBMRegressor(
    n_estimators=2000,
    learning_rate=0.02,
    num_leaves=63,
    min_child_samples=50,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=0.1,
    random_state=42,
    verbose=-1,
)
model_reg.fit(
    X_train2, y_train2,
    eval_set=[(X_test2, y_test2)],
    callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(period=-1)]
)

pred2 = model_reg.predict(X_test2)
rmse = np.sqrt(mean_squared_error(y_test2, pred2))
# 예측값의 분산 확인
print(f"  ✅ RMSE: {rmse:.2f}분")
print(f"  예측값 분포: min={pred2.min():.1f}, mean={pred2.mean():.1f}, max={pred2.max():.1f}")

# 수술 시간 잔차 표준편차 (불확실성 추정용)
resid_std = np.std(y_test2.values - pred2)
print(f"  예측 잔차 표준편차: {resid_std:.1f}분")

# 저장
print("\n  모델 저장 중...")
joblib.dump(model_cls,  'model_cls.pkl')
joblib.dump(model_reg,  'model_reg.pkl')
joblib.dump(encoders,   'encoders.pkl')
joblib.dump(medians,    'medians.pkl')
joblib.dump(features,   'features.pkl')
joblib.dump(resid_std,  'resid_std.pkl')   # 예측 불확실성

# ── 혈액검사 입력 → 실제 피처명 매핑 저장 ──────────────────────
# 실제 학습된 피처 중 혈액검사 관련 컬럼명을 추출해서 저장
lab_feature_map = {}
keywords = {
    'lab_creatinine':  ['creatinine', 'creat'],
    'lab_glucose':     ['glucose', 'glu'],
    'lab_hb':          ['hb', 'hemoglobin', 'haemoglobin'],
    'lab_albumin':     ['albumin', 'alb'],
    'lab_wbc':         ['wbc', 'leukocyte', 'white'],
    'lab_plt':         ['plt', 'platelet'],
    'lab_na':          ['sodium', 'na'],
    'lab_k':           ['potassium', 'kalium', '_k'],
    'lab_alt':         ['alt', 'gpt'],
    'lab_ast':         ['ast', 'got'],
    'lab_bun':         ['bun', 'urea', 'urea_nitrogen'],
    'lab_crp':         ['crp', 'c_reactive'],
    'lab_inr':         ['inr', 'pt_inr'],
    'lab_tbili':       ['tbili', 'total_bili', 'bilirubin'],
    'lab_hct':         ['hct', 'hematocrit'],
    'lab_calcium':     ['calcium', 'ca'],
    'lab_pt':          ['_pt_', 'prothrombin'],
    'lab_aptt':        ['aptt', 'ptt'],
}

for input_key, kws in keywords.items():
    matched = []
    for feat in features:
        feat_lower = feat.lower()
        if any(kw.lower() in feat_lower for kw in kws):
            matched.append(feat)
    if matched:
        lab_feature_map[input_key] = matched

joblib.dump(lab_feature_map, 'lab_feature_map.pkl')
print(f'혈액검사 매핑 저장: {len(lab_feature_map)}개 항목')
for k, v in lab_feature_map.items():
    print(f'  {k} -> {v}')

print(f"\n{'='*60}")
print(f"  완료! AUROC {auroc:.4f} | RMSE {rmse:.2f}분")
print(f"  저장 파일: model_cls.pkl, model_reg.pkl, encoders.pkl,")
print(f"             medians.pkl, features.pkl, resid_std.pkl")
print(f"{'='*60}")

# ── 수술 코드 목록 저장 (프론트엔드 드롭다운용) ─────────────────
print("\n수술 코드 목록 저장 중...")

# 진료과별 상위 수술 코드 추출
icd10_pcs_lookup = (
    ops.groupby(['department', 'icd10_pcs'])
    .agg(count=('opid' if 'opid' in ops.columns else 'subject_id', 'count'),
         avg_duration=('surgery_duration', 'mean'))
    .reset_index()
    .sort_values('count', ascending=False)
)

# 진료과 역매핑 (인코딩된 숫자 → 원래 문자열)
dept_le = encoders['department']
pcs_le  = encoders['icd10_pcs']
icd10_pcs_lookup['department_name'] = dept_le.inverse_transform(icd10_pcs_lookup['department'].astype(int))
icd10_pcs_lookup['icd10_pcs_code']  = pcs_le.inverse_transform(icd10_pcs_lookup['icd10_pcs'].astype(int))

# 진료과별 상위 30개만
top_surgeries = (
    icd10_pcs_lookup
    .groupby('department_name')
    .apply(lambda x: x.nlargest(30, 'count'))
    .reset_index(drop=True)
)[['department_name', 'icd10_pcs_code', 'count', 'avg_duration']]

top_surgeries['avg_duration'] = top_surgeries['avg_duration'].round(1)
joblib.dump(top_surgeries, 'icd10_pcs_lookup.pkl')
print(f"✅ 수술 코드 저장 완료: {len(top_surgeries)}개 (진료과별 상위 30개)")
print(f"   샘플:\n{top_surgeries.head(10).to_string(index=False)}")
