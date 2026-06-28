import re
import math
import os
import json
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import joblib
import numpy as np
import pandas as pd
import xgboost as xgb  # XGBoost 추가

app = Flask(__name__)
CORS(app)

print("=" * 55)
print("  SurgicalAI 서버 로딩 중... (XGBoost 통합 버전)")
print("=" * 55)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 1. 기존 데이터 및 모델 로드
model_reg = joblib.load(os.path.join(BASE_DIR, 'model_reg.pkl'))
encoders  = joblib.load(os.path.join(BASE_DIR, 'encoders.pkl'))
medians   = joblib.load(os.path.join(BASE_DIR, 'medians.pkl'))
features  = joblib.load(os.path.join(BASE_DIR, 'features.pkl')) # 기존 전체 피처

try:    resid_std  = float(joblib.load(os.path.join(BASE_DIR, 'resid_std.pkl')))
except: resid_std  = 45.0

try:
    pcs_lookup = joblib.load(os.path.join(BASE_DIR, 'icd10_pcs_lookup.pkl'))
    print(f"✅ 수술 코드 로드: {len(pcs_lookup)}개")
except Exception as e:
    pcs_lookup = None
    print(f"⚠ icd10_pcs_lookup.pkl 없음: {e}")

# 2. 🌟 새로운 XGBoost 분류 모델 및 피처 컬럼 로드 🌟
try:
    xgb_model = xgb.Booster()
    xgb_model.load_model(os.path.join(BASE_DIR, 'xgboost_high_risk_model_best_threshold.json'))
    
    with open(os.path.join(BASE_DIR, 'feature_columns.json'), 'r', encoding='utf-8') as f:
        xgb_features = json.load(f)
    print("✅ XGBoost 분류 모델 및 피처 컬럼 로드 완료")
except Exception as e:
    print(f"⚠ XGBoost 모델 로딩 실패: {e}")
    xgb_model = None
    xgb_features = []

# 인코더 안전 변환 함수
def safe_encode(col, val):
    le = encoders.get(col)
    if le is None: return 0
    try: return int(le.transform([str(val)])[0])
    except: return 0

# (기존에 쓰시던 FEATURE_KR 매핑, format_value, get_clinical_notes 함수 등은 그대로 유지했다고 가정합니다.
# 코드 길이를 위해 생략된 부분이 있다면 기존 코드의 함수들을 이 자리에 그대로 두시면 됩니다.)
def to_kr(col):
    # 기존에 쓰시던 한글 매핑 로직 유지
    return col

def fmt_val(col, val):
    # 기존에 쓰시던 포맷팅 로직 유지
    return round(float(val), 2)

@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

@app.route('/surgeries')
def get_surgeries():
    dept = request.args.get('dept')
    if pcs_lookup is None: return jsonify([])
    filtered = pcs_lookup[pcs_lookup['department_name'] == dept]
    res = []
    for _, row in filtered.iterrows():
        res.append({
            'code': row['icd10_pcs_code'],
            'label': f"{row['icd10_pcs_code']} (평균 {row['avg_duration']:.0f}분, {int(row['count'])}건)"
        })
    return jsonify(res)

@app.route('/predict', methods=['POST'])
def predict():
    d = request.json
    icd_code = d.get('surgery', '')
    
    # 기존 medians 전체 피처로 기본 row 생성 (KeyError 방지)
    row = {f: medians.get(f, 0) for f in features}
    
    # 사용자 입력 덮어쓰기
    row.update({
        'age': float(d.get('age', 50)),
        'weight': float(d.get('weight', 65)),
        'height': float(d.get('height', 165)),
        'asa': float(d.get('asa', 2)),
        'emop': float(d.get('emop', 0)),
        'diag_count': float(d.get('diag_count', 0)),
        'sex': safe_encode('sex', d.get('sex', 'M')),
        'department': safe_encode('department', d.get('dept', 'GS')),
        'antype': safe_encode('antype', d.get('antype', 'General')),
        'icd10_pcs': safe_encode('icd10_pcs', icd_code),
        'surgery_rank': 1.0 # 첫 수술 가정
    })
    
    # ── [1] 위험도 예측 (XGBoost 사용) ──
    # 형님이 주신 feature_columns.json 순서에 정확히 맞춰서 데이터프레임 생성
    X_xgb = pd.DataFrame([row])[xgb_features]
    
    # XGBoost 전용 DMatrix 변환
    dtest = xgb.DMatrix(X_xgb)
    
    # 예측 수행 (XGBoost는 바로 확률(Probability) 또는 Margin을 반환함)
    raw_prob = float(xgb_model.predict(dtest)[0])
    icu_prob = raw_prob * 100

    # XGBoost 변수 중요도 추출 (LightGBM과 방식이 다름)
    xgb_imp_dict = xgb_model.get_score(importance_type='gain')
    imp_cls = np.array([xgb_imp_dict.get(f, 0.0) for f in xgb_features])
    
    top_cls = np.argsort(imp_cls)[::-1][:8]
    p80_cls = np.percentile(imp_cls[imp_cls > 0], 80) if len(imp_cls[imp_cls > 0]) > 0 else 0
    p50_cls = np.percentile(imp_cls[imp_cls > 0], 50) if len(imp_cls[imp_cls > 0]) > 0 else 0

    risk_factors = []
    for i in top_cls:
        col = xgb_features[i] # XGBoost용 피처 이름 사용
        imp = imp_cls[i]
        impact = 'high' if imp >= p80_cls else ('mid' if imp >= p50_cls else 'ok')
        risk_factors.append({
            'name':   to_kr(col),
            'value':  fmt_val(col, row[col]),
            'impact': impact
        })

    # ── [2] 수술 시간 예측 (기존 LightGBM 유지) ──
    X_reg = pd.DataFrame([row])[features]
    surgery_min = float(model_reg.predict(X_reg)[0])
    surgery_min = max(5.0, min(1440.0, surgery_min))
    
    surgery_fast = max(5.0, surgery_min - resid_std * 0.7)
    surgery_slow = min(1440.0, surgery_min + resid_std * 1.2)

    imp_reg = model_reg.feature_importances_
    top_reg = np.argsort(imp_reg)[::-1][:8]
    p80_reg = np.percentile(imp_reg[imp_reg > 0], 80) if len(imp_reg[imp_reg > 0]) > 0 else 0
    p50_reg = np.percentile(imp_reg[imp_reg > 0], 50) if len(imp_reg[imp_reg > 0]) > 0 else 0

    time_factors = []
    for i in top_reg:
        col = features[i]
        imp = imp_reg[i]
        impact = 'high' if imp >= p80_reg else ('mid' if imp >= p50_reg else 'ok')
        time_factors.append({
            'name':   to_kr(col),
            'value':  fmt_val(col, row[col]),
            'impact': impact
        })

    risk_level = '고위험' if icu_prob >= 50 else ('중위험' if icu_prob >= 20 else '저위험')
    recommendation = (
        f"해당 환자는 ICU 입원 확률 {icu_prob:.1f}%로 {risk_level}군으로 분류됩니다. "
        f"예상 수술 시간은 약 {surgery_min:.0f}분이며, "
        f"빠르면 {surgery_fast:.0f}분, 늦으면 {surgery_slow:.0f}분 예상됩니다. "
    )

    return jsonify({
        'icu_prob':       round(icu_prob, 1),
        'surgery_min':    round(surgery_min),
        'surgery_fast':   round(surgery_fast),
        'surgery_slow':   round(surgery_slow),
        'risk_factors':   risk_factors,
        'time_factors':   time_factors,
        'recommendation': recommendation,
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)