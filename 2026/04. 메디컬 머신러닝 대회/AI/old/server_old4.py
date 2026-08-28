from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)
CORS(app)

print("=" * 50)
print("  모델 로딩 중...")
print("=" * 50)
model_cls = joblib.load('model_cls.pkl')
model_reg = joblib.load('model_reg.pkl')
encoders  = joblib.load('encoders.pkl')
medians   = joblib.load('medians.pkl')
features  = joblib.load('features.pkl')

try:
    resid_std = float(joblib.load('resid_std.pkl'))
except:
    resid_std = 40.0

try:
    pcs_lookup = joblib.load('icd10_pcs_lookup.pkl')
except:
    pcs_lookup = None
    print("⚠ icd10_pcs_lookup.pkl 없음")

try:
    lab_feature_map = joblib.load('lab_feature_map.pkl')
    print(f"✅ 혈액검사 매핑 로드: {len(lab_feature_map)}개")
    for k, v in lab_feature_map.items():
        print(f"   {k} -> {v[:2]}{'...' if len(v)>2 else ''}")
except:
    lab_feature_map = {}
    print("⚠ lab_feature_map.pkl 없음 — train.py 재실행 필요")

print(f"✅ 피처 수: {len(features)}개")
print(f"✅ 예측 불확실성: ±{resid_std:.1f}분")

# ── 한글 라벨 ────────────────────────────────────────────────
FEATURE_KR = {
    'age': '나이', 'sex': '성별', 'weight': '체중', 'height': '신장',
    'asa': 'ASA 점수 (건강상태)', 'emop': '응급수술 여부',
    'department': '진료과', 'antype': '마취 유형',
    'icd10_pcs': '수술 종류 코드', 'diag_count': '기저 진단 수',
    'surgery_rank': '수술 횟수',
    'diag_circulatory':     '심혈관 질환',
    'diag_neoplasm':        '종양 / 암',
    'diag_endocrine':       '당뇨 / 내분비 질환',
    'diag_respiratory':     '호흡기 질환',
    'diag_digestive':       '소화기 질환',
    'diag_musculoskeletal': '근골격계 질환',
    'diag_genitourinary':   '비뇨생식기 질환',
    'diag_injury':          '외상 / 손상',
    'diag_nervous':         '신경계 질환',
    'diag_mental':          '정신건강 질환',
    'mean_bt':'평균 체온', 'std_bt':'체온 변동성', 'last_bt':'최근 체온',
    'mean_hr':'평균 심박수', 'std_hr':'심박수 변동성', 'last_hr':'최근 심박수',
    'mean_rr':'평균 호흡수', 'std_rr':'호흡수 변동성',
    'mean_spo2':'평균 산소포화도', 'min_spo2':'최저 산소포화도',
    'mean_nibp_sbp':'평균 수축기 혈압', 'std_nibp_sbp':'수축기 혈압 변동성',
    'mean_nibp_dbp':'평균 이완기 혈압', 'std_nibp_dbp':'이완기 혈압 변동성',
    'mean_nibp_mbp':'평균 평균동맥압',
    'mean_uo':'평균 소변량', 'std_uo':'소변량 변동성',
}

# 혈액검사 한글 라벨 (입력키 → 한글)
LAB_KR = {
    'lab_creatinine': '크레아티닌 (신장 기능)',
    'lab_glucose':    '혈당',
    'lab_hb':         '혈색소 (빈혈 지표)',
    'lab_albumin':    '알부민 (영양 상태)',
    'lab_wbc':        '백혈구 수 (면역/염증)',
    'lab_plt':        '혈소판 수',
    'lab_na':         '혈중 나트륨',
    'lab_k':          '혈중 칼륨',
    'lab_alt':        '간수치 ALT',
    'lab_ast':        '간수치 AST',
    'lab_bun':        '혈액요소질소 (신장)',
    'lab_crp':        'C반응성 단백 (염증)',
    'lab_inr':        '혈액응고 INR',
    'lab_tbili':      '총 빌리루빈 (간 기능)',
    'lab_hct':        '적혈구 용적률',
    'lab_calcium':    '혈중 칼슘',
    'lab_pt':         '프로트롬빈 시간',
    'lab_aptt':       '혈액응고 시간 APTT',
}

# 피처명 → 한글 (lab 포함)
def feat_to_kr(feat):
    if feat in FEATURE_KR:
        return FEATURE_KR[feat]
    # lab 피처 역매핑
    for input_key, feat_list in lab_feature_map.items():
        if feat in feat_list:
            return LAB_KR.get(input_key, feat)
    return feat.replace('_', ' ')

# 단위 포매팅
LAB_UNITS = {
    'lab_creatinine': 'mg/dL', 'lab_glucose': 'mg/dL',
    'lab_hb': 'g/dL', 'lab_albumin': 'g/dL',
    'lab_wbc': '×10³/μL', 'lab_plt': '×10³/μL',
    'lab_na': 'mEq/L', 'lab_k': 'mEq/L',
    'lab_alt': 'IU/L', 'lab_ast': 'IU/L',
    'lab_bun': 'mg/dL', 'lab_crp': 'mg/L',
    'lab_inr': '', 'lab_tbili': 'mg/dL',
}
VITAL_UNITS = {
    'age':'세', 'weight':'kg', 'height':'cm',
    'mean_bt':'°C', 'last_bt':'°C',
    'mean_hr':'회/분', 'last_hr':'회/분',
    'mean_rr':'회/분', 'mean_spo2':'%', 'min_spo2':'%',
    'mean_nibp_sbp':'mmHg', 'mean_nibp_dbp':'mmHg', 'mean_nibp_mbp':'mmHg',
    'mean_uo':'mL', 'diag_count':'개',
}

def fmt_val(feat, val):
    val = float(val)
    if feat in ['emop']:
        return '응급' if val >= 0.5 else '계획수술'
    if feat.startswith('diag_') and feat != 'diag_count':
        return '있음' if val >= 0.5 else '없음'
    # lab 단위
    for input_key, feat_list in lab_feature_map.items():
        if feat in feat_list:
            unit = LAB_UNITS.get(input_key, '')
            return f"{val:.2f}{unit}" if val != int(val) else f"{int(val)}{unit}"
    unit = VITAL_UNITS.get(feat, '')
    return f"{val:.1f}{unit}" if val != int(val) else f"{int(val)}{unit}"

# 진료과 매핑
DEPT_MAP = {
    'GS':'GS','CTS':'CTS','NS':'NS','OS':'OS',
    'OG':'OG','UR':'UR','PS':'PS','OL':'OL',
    '일반외과':'GS','흉부외과':'CTS','신경외과':'NS','정형외과':'OS',
    '산부인과':'OG','비뇨기과':'UR','성형외과':'PS','안과':'OL',
}
DEPT_NAME = {v:k for k,v in DEPT_MAP.items() if len(k)>2}

def safe_encode(col, val):
    le = encoders.get(col)
    if le is None:
        return medians.get(col, 0)
    try:
        return int(le.transform([str(val)])[0])
    except ValueError:
        return medians.get(col, 0)

# ── /surgeries 엔드포인트 ────────────────────────────────────
@app.route('/surgeries')
def get_surgeries():
    if pcs_lookup is None:
        return jsonify([])
    dept = request.args.get('dept', '')
    dept_name = DEPT_NAME.get(dept, '')
    if dept_name:
        filtered = pcs_lookup[pcs_lookup['department_name'] == dept_name]
    else:
        filtered = pcs_lookup
    result = []
    for _, row in filtered.iterrows():
        result.append({
            'code':  row['icd10_pcs_code'],
            'label': f"{row['icd10_pcs_code']}  (평균 {row['avg_duration']:.0f}분 · {int(row['count']):,}건)",
            'avg_duration': row['avg_duration'],
        })
    return jsonify(result)

@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

# ── /predict 엔드포인트 ──────────────────────────────────────
@app.route('/predict', methods=['POST'])
def predict():
    d = request.json

    # 기본값: 학습 중앙값
    row = {f: medians.get(f, 0) for f in features}

    # 기본 정보
    row['age']          = float(d.get('age') or medians.get('age', 55))
    row['weight']       = float(d.get('weight') or medians.get('weight', 65))
    row['height']       = float(d.get('height') or medians.get('height', 165))
    row['asa']          = float(d.get('asa') or medians.get('asa', 2))
    row['emop']         = float(d.get('emop', 0))
    row['diag_count']   = float(d.get('diag_count') or 0)
    row['surgery_rank'] = 1.0
    row['sex']          = safe_encode('sex',    d.get('sex', 'M'))
    row['race']         = safe_encode('race',   'Korean')

    dept_code = DEPT_MAP.get(d.get('dept', ''), '')
    row['department'] = safe_encode('department', dept_code)
    row['antype']     = safe_encode('antype',     d.get('antype', 'General'))

    # ── 핵심: 수술 코드 — 실제 ICD-10-PCS 코드 직접 사용 ──────
    icd10_pcs_code = d.get('icd10_pcs_code', '')
    row['icd10_pcs'] = safe_encode('icd10_pcs', icd10_pcs_code)

    # 진단명 (질환군 플래그)
    for col in ['diag_circulatory','diag_neoplasm','diag_endocrine',
                'diag_respiratory','diag_digestive','diag_musculoskeletal',
                'diag_genitourinary','diag_injury','diag_nervous','diag_mental']:
        if col in features:
            row[col] = float(d.get(col, 0))

    # ── 핵심: 혈액검사 수치 — lab_feature_map 기반 정확한 매핑 ──
    lab_inputs_used = []  # 실제로 사용된 입력 추적
    lab_input_keys = [
        'lab_creatinine','lab_glucose','lab_hb','lab_albumin',
        'lab_wbc','lab_plt','lab_na','lab_k',
        'lab_alt','lab_ast','lab_bun','lab_crp',
        'lab_inr','lab_tbili','lab_hct','lab_calcium','lab_pt','lab_aptt',
    ]
    for input_key in lab_input_keys:
        val = d.get(input_key)
        if val is None or val == '':
            continue
        val = float(val)
        matched_feats = lab_feature_map.get(input_key, [])
        for feat in matched_feats:
            if feat in features:
                row[feat] = val
                lab_inputs_used.append((input_key, feat, val))

    if lab_inputs_used:
        print(f"  혈액검사 입력 반영: {len(lab_inputs_used)}개")
        for inp, feat, val in lab_inputs_used[:5]:
            print(f"    {inp}={val} → {feat}")

    # ── 예측 ────────────────────────────────────────────────────
    X = pd.DataFrame([row])[features]
    icu_prob    = float(model_cls.predict_proba(X)[0][1]) * 100
    surgery_min = float(model_reg.predict(X)[0])
    surgery_min = max(5.0, min(1440.0, surgery_min))
    surgery_fast = max(5.0, surgery_min - resid_std * 0.7)
    surgery_slow = min(1440.0, surgery_min + resid_std * 1.2)

    # ── 위험 요인 추출 ───────────────────────────────────────────
    imp_cls = model_cls.feature_importances_
    imp_reg = model_reg.feature_importances_

    # 사용자가 입력한 값이 있는 피처 우선 + 중요도 상위 순
    user_input_feats = set()
    for _, feat, _ in lab_inputs_used:
        user_input_feats.add(feat)
    for col in ['diag_circulatory','diag_neoplasm','diag_endocrine',
                'diag_respiratory','diag_digestive','diag_musculoskeletal',
                'diag_genitourinary','diag_injury']:
        if d.get(col, 0) == 1:
            user_input_feats.add(col)

    # 피처 → 입력키 역매핑 (중복 제거용)
    feat_to_input_key = {}
    for input_key, feat_list in lab_feature_map.items():
        for feat in feat_list:
            feat_to_input_key[feat] = input_key

    def get_top_factors(importances, n=7):
        idx_sorted = np.argsort(importances)[::-1]
        factors = []
        seen_feats = set()
        seen_input_keys = set()  # 같은 혈액검사 항목 중복 방지

        # 1. 사용자 입력 피처 먼저 (입력키당 1개만)
        for i in idx_sorted:
            feat = features[i]
            input_key = feat_to_input_key.get(feat)
            if feat in user_input_feats:
                if input_key and input_key in seen_input_keys:
                    continue  # 같은 혈액검사 항목 중복 스킵
                factors.append(i)
                seen_feats.add(feat)
                if input_key:
                    seen_input_keys.add(input_key)

        # 2. 나머지 중요도 순 (입력키 중복 방지)
        for i in idx_sorted:
            feat = features[i]
            if feat in seen_feats:
                continue
            input_key = feat_to_input_key.get(feat)
            if input_key and input_key in seen_input_keys:
                continue
            if len(factors) < n:
                factors.append(i)
                seen_feats.add(feat)
                if input_key:
                    seen_input_keys.add(input_key)
            if len(factors) >= n:
                break
        return factors

    p80_cls = np.percentile(imp_cls[imp_cls > 0], 80)
    p50_cls = np.percentile(imp_cls[imp_cls > 0], 50)
    p80_reg = np.percentile(imp_reg[imp_reg > 0], 80)
    p50_reg = np.percentile(imp_reg[imp_reg > 0], 50)

    risk_factors = []
    for i in get_top_factors(imp_cls):
        feat = features[i]
        imp  = imp_cls[i]
        impact = 'high' if imp >= p80_cls else ('mid' if imp >= p50_cls else 'ok')
        # 사용자가 직접 입력한 값이면 강조
        is_user = feat in user_input_feats
        risk_factors.append({
            'name':    feat_to_kr(feat),
            'value':   fmt_val(feat, row[feat]),
            'impact':  impact,
            'is_user': is_user,
        })

    time_factors = []
    for i in get_top_factors(imp_reg):
        feat = features[i]
        imp  = imp_reg[i]
        impact = 'high' if imp >= p80_reg else ('mid' if imp >= p50_reg else 'ok')
        is_user = feat in user_input_feats
        time_factors.append({
            'name':    feat_to_kr(feat),
            'value':   fmt_val(feat, row[feat]),
            'impact':  impact,
            'is_user': is_user,
        })

    risk_level = '고위험' if icu_prob >= 50 else ('중위험' if icu_prob >= 20 else '저위험')
    recommendation = (
        f"해당 환자는 ICU 입원 확률 {icu_prob:.1f}%로 {risk_level}군으로 분류됩니다. "
        f"예상 수술 시간은 약 {surgery_min:.0f}분이며, "
        f"빠르면 {surgery_fast:.0f}분, 늦으면 {surgery_slow:.0f}분 내에 종료될 것으로 예측됩니다. "
        + ('수술 전 ICU 준비 및 집중 모니터링을 권고합니다.' if icu_prob >= 50
           else '표준 수술 프로토콜에 따라 진행하며 활력징후를 지속 관찰하십시오.')
    )

    return jsonify({
        'icu_prob':       round(icu_prob, 1),
        'surgery_min':    round(surgery_min),
        'surgery_fast':   round(surgery_fast),
        'surgery_slow':   round(surgery_slow),
        'risk_factors':   risk_factors,
        'time_factors':   time_factors,
        'recommendation': recommendation,
        'lab_count':      len(lab_inputs_used),
    })

if __name__ == '__main__':
    print("http://localhost:5000 에서 실행 중!")
    app.run(debug=True, port=5000)
