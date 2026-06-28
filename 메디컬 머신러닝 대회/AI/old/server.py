from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)
CORS(app)

print("모델 로딩 중...")
model_cls = joblib.load('model_cls.pkl')
model_reg = joblib.load('model_reg.pkl')
encoders  = joblib.load('encoders.pkl')
medians   = joblib.load('medians.pkl')
features  = joblib.load('features.pkl')
print("모델 로드 완료!")

# ── 컬럼명 한글 변환 ──────────────────────────────────────────
FEATURE_KR = {
    'age':              '나이',
    'sex':              '성별',
    'weight':           '체중',
    'height':           '신장',
    'asa':              'ASA 점수 (건강상태)',
    'emop':             '응급수술 여부',
    'department':       '진료과',
    'antype':           '마취 유형',
    'icd10_pcs':        '수술 코드 (수술 종류)',
    'diag_count':       '기저 진단 수',
    'mean_bt':          '평균 체온',
    'min_bt':           '최저 체온',
    'max_bt':           '최고 체온',
    'std_bt':           '체온 변동성',
    'mean_hr':          '평균 심박수',
    'min_hr':           '최저 심박수',
    'max_hr':           '최고 심박수',
    'std_hr':           '심박수 변동성',
    'mean_rr':          '평균 호흡수',
    'min_rr':           '최저 호흡수',
    'max_rr':           '최고 호흡수',
    'std_rr':           '호흡수 변동성',
    'mean_spo2':        '평균 산소포화도',
    'min_spo2':         '최저 산소포화도',
    'max_spo2':         '최고 산소포화도',
    'std_spo2':         '산소포화도 변동성',
    'mean_nibp_sbp':    '평균 수축기 혈압',
    'min_nibp_sbp':     '최저 수축기 혈압',
    'max_nibp_sbp':     '최고 수축기 혈압',
    'std_nibp_sbp':     '수축기 혈압 변동성',
    'mean_nibp_dbp':    '평균 이완기 혈압',
    'min_nibp_dbp':     '최저 이완기 혈압',
    'max_nibp_dbp':     '최고 이완기 혈압',
    'std_nibp_dbp':     '이완기 혈압 변동성',
    'mean_nibp_mbp':    '평균 평균동맥압',
    'min_nibp_mbp':     '최저 평균동맥압',
    'max_nibp_mbp':     '최고 평균동맥압',
    'std_nibp_mbp':     '평균동맥압 변동성',
    'mean_uo':          '평균 소변량',
    'min_uo':           '최저 소변량',
    'max_uo':           '최고 소변량',
    'std_uo':           '소변량 변동성',
    'mean_fio2':        '평균 산소농도 (FiO2)',
    'min_fio2':         '최저 산소농도',
    'max_fio2':         '최고 산소농도',
    'std_fio2':         '산소농도 변동성',
    'mean_gcs_e':       '평균 GCS 눈반응',
    'min_gcs_e':        '최저 GCS 눈반응',
    'max_gcs_e':        '최고 GCS 눈반응',
    'std_gcs_e':        'GCS 눈반응 변동성',
    'mean_gcs_m':       '평균 GCS 운동반응',
    'min_gcs_m':        '최저 GCS 운동반응',
    'max_gcs_m':        '최고 GCS 운동반응',
    'std_gcs_m':        'GCS 운동반응 변동성',
    'mean_gcs_v':       '평균 GCS 언어반응',
    'min_gcs_v':        '최저 GCS 언어반응',
    'max_gcs_v':        '최고 GCS 언어반응',
    'std_gcs_v':        'GCS 언어반응 변동성',
    'mean_vent':        '인공호흡기 사용률',
    'min_vent':         '인공호흡기 최솟값',
    'max_vent':         '인공호흡기 최댓값',
    'std_vent':         '인공호흡기 변동성',
    'mean_crrt':        'CRRT (신장대체요법) 평균',
    'min_crrt':         'CRRT 최솟값',
    'max_crrt':         'CRRT 최댓값',
    'std_crrt':         'CRRT 변동성',
    'mean_ecmo':        'ECMO (체외순환) 평균',
    'min_ecmo':         'ECMO 최솟값',
    'max_ecmo':         'ECMO 최댓값',
    'std_ecmo':         'ECMO 변동성',
    'mean_iabp':        '대동맥내풍선펌프 평균',
    'min_iabp':         '대동맥내풍선펌프 최솟값',
    'max_iabp':         '대동맥내풍선펌프 최댓값',
    'std_iabp':         '대동맥내풍선펌프 변동성',
}

def to_kr(col):
    return FEATURE_KR.get(col, col)

# ── 값 한글 변환 ──────────────────────────────────────────────
def format_value(col, val):
    val = round(float(val), 2)
    units = {
        'age': '세', 'weight': 'kg', 'height': 'cm',
        'mean_bt': '°C', 'min_bt': '°C', 'max_bt': '°C', 'std_bt': '°C',
        'mean_hr': '회/분', 'min_hr': '회/분', 'max_hr': '회/분',
        'mean_rr': '회/분', 'min_rr': '회/분', 'max_rr': '회/분',
        'mean_spo2': '%', 'min_spo2': '%',
        'mean_nibp_sbp': 'mmHg', 'max_nibp_sbp': 'mmHg',
        'mean_nibp_dbp': 'mmHg', 'min_nibp_dbp': 'mmHg',
        'mean_nibp_mbp': 'mmHg',
        'mean_uo': 'mL', 'max_uo': 'mL',
        'diag_count': '개',
    }
    unit = units.get(col, '')
    if col == 'emop':
        return '응급' if val == 1 else '계획수술'
    if col == 'sex':
        return '남성' if val == 0 else '여성'
    return f"{val}{unit}"

DEPT_MAP = {
    "일반외과 (General Surgery)": "GS",
    "흉부외과 (Cardio-Thoracic)": "CTS",
    "신경외과 (Neurosurgery)":    "NS",
    "정형외과 (Orthopedic)":      "OS",
    "산부인과 (Ob/Gyn)":          "OG",
    "비뇨기과 (Urology)":         "UR",
    "성형외과 (Plastic Surgery)": "PS",
    "안과 (Ophthalmology)":       "OL",
}

def safe_encode(col, val):
    le = encoders.get(col)
    if le is None:
        return 0
    try:
        return int(le.transform([val])[0])
    except ValueError:
        return 0

@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

@app.route('/predict', methods=['POST'])
def predict():
    d = request.json
    dept_code = DEPT_MAP.get(d.get('dept', ''), d.get('dept', ''))

    row = {f: medians.get(f, 0) for f in features}
    row['age']        = float(d.get('age', 50))
    row['weight']     = float(d.get('weight', 65))
    row['height']     = float(d.get('height', 165))
    row['asa']        = float(d.get('asa', 2))
    row['emop']       = float(d.get('emop', 0))
    row['diag_count'] = float(d.get('diag_count', 1))
    row['sex']        = safe_encode('sex',        d.get('sex', 'M'))
    row['race']       = safe_encode('race',       'Korean')
    row['department'] = safe_encode('department', dept_code)
    row['antype']     = safe_encode('antype',     d.get('antype', 'General'))
    row['icd10_pcs']  = safe_encode('icd10_pcs',  d.get('icd10_pcs', ''))

    X = pd.DataFrame([row])[features]

    icu_prob    = float(model_cls.predict_proba(X)[0][1]) * 100
    surgery_min = max(10.0, float(model_reg.predict(X)[0]))

    imp_cls = model_cls.feature_importances_
    top_cls = np.argsort(imp_cls)[::-1][:5]
    p80_cls = np.percentile(imp_cls, 80)
    p50_cls = np.percentile(imp_cls, 50)

    risk_factors = []
    for i in top_cls:
        col = features[i]
        imp = imp_cls[i]
        impact = 'high' if imp > p80_cls else ('mid' if imp > p50_cls else 'ok')
        risk_factors.append({
            'name':   to_kr(col),
            'value':  format_value(col, row[col]),
            'impact': impact
        })

    imp_reg = model_reg.feature_importances_
    top_reg = np.argsort(imp_reg)[::-1][:5]
    p80_reg = np.percentile(imp_reg, 80)
    p50_reg = np.percentile(imp_reg, 50)

    time_factors = []
    for i in top_reg:
        col = features[i]
        imp = imp_reg[i]
        impact = 'high' if imp > p80_reg else ('mid' if imp > p50_reg else 'ok')
        time_factors.append({
            'name':   to_kr(col),
            'value':  format_value(col, row[col]),
            'impact': impact
        })

    risk_level = '고위험' if icu_prob >= 50 else ('중위험' if icu_prob >= 20 else '저위험')
    recommendation = (
        f"해당 환자는 ICU 입원 확률 {icu_prob:.1f}%로 {risk_level}군으로 분류됩니다. "
        f"예상 수술 시간은 약 {surgery_min:.0f}분이며, "
        f"{'수술 전 ICU 사전 준비 및 면밀한 모니터링이 권고됩니다.' if icu_prob >= 50 else '표준 수술 프로토콜에 따라 진행하되 활력징후를 지속 관찰하십시오.'}"
    )

    return jsonify({
        'icu_prob':       round(icu_prob, 1),
        'surgery_min':    round(surgery_min),
        'risk_factors':   risk_factors,
        'time_factors':   time_factors,
        'recommendation': recommendation,
    })

if __name__ == '__main__':
    print("http://localhost:5000 접속하면 앱 열려!")
    app.run(debug=True, port=5000)
