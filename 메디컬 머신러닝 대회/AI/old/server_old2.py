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
try:
    resid_std = float(joblib.load('resid_std.pkl'))
except:
    resid_std = 40.0
try:
    pcs_lookup = joblib.load('icd10_pcs_lookup.pkl')
except:
    pcs_lookup = None
    print('⚠ icd10_pcs_lookup.pkl 없음 - train.py 재실행 필요')
print(f"✅ 로드 완료 | 피처: {len(features)}개 | 예측 불확실성: ±{resid_std:.1f}분")

# ── 한글 라벨 ────────────────────────────────────────────────
FEATURE_KR = {
    'age': '나이', 'sex': '성별', 'weight': '체중', 'height': '신장',
    'asa': 'ASA 점수 (건강상태)', 'emop': '응급수술 여부',
    'department': '진료과', 'antype': '마취 유형',
    'icd10_pcs': '수술 코드 (수술 종류)', 'diag_count': '기저 진단 수',
    'surgery_rank': '수술 횟수 (누적)',
    # 질환군
    'diag_circulatory':     '심혈관 질환',
    'diag_neoplasm':        '종양/암',
    'diag_endocrine':       '당뇨/내분비 질환',
    'diag_respiratory':     '호흡기 질환',
    'diag_digestive':       '소화기 질환',
    'diag_musculoskeletal': '근골격계 질환',
    'diag_genitourinary':   '비뇨생식기 질환',
    'diag_injury':          '외상/손상',
    'diag_nervous':         '신경계 질환',
    'diag_mental':          '정신건강 질환',
    # 생체징후
    'mean_bt':'평균 체온', 'min_bt':'최저 체온', 'max_bt':'최고 체온', 'std_bt':'체온 변동성', 'last_bt':'최근 체온',
    'mean_hr':'평균 심박수', 'min_hr':'최저 심박수', 'max_hr':'최고 심박수', 'std_hr':'심박수 변동성', 'last_hr':'최근 심박수',
    'mean_rr':'평균 호흡수', 'min_rr':'최저 호흡수', 'max_rr':'최고 호흡수', 'std_rr':'호흡수 변동성',
    'mean_spo2':'평균 산소포화도', 'min_spo2':'최저 산소포화도', 'last_spo2':'최근 산소포화도',
    'mean_nibp_sbp':'평균 수축기 혈압', 'min_nibp_sbp':'최저 수축기 혈압', 'max_nibp_sbp':'최고 수축기 혈압',
    'mean_nibp_dbp':'평균 이완기 혈압', 'min_nibp_dbp':'최저 이완기 혈압', 'std_nibp_dbp':'이완기 혈압 변동성',
    'mean_nibp_mbp':'평균 평균동맥압', 'std_nibp_mbp':'평균동맥압 변동성',
    'mean_uo':'평균 소변량', 'max_uo':'최대 소변량', 'std_uo':'소변량 변동성',
    'mean_vent':'인공호흡기 사용',
    # 혈액검사 - mean
    'lab_mean_creatinine':'크레아티닌 (신장 기능)', 'lab_last_creatinine':'최근 크레아티닌',
    'lab_mean_glucose':'혈당', 'lab_last_glucose':'최근 혈당',
    'lab_mean_hb':'혈색소 (빈혈)', 'lab_last_hb':'최근 혈색소',
    'lab_mean_albumin':'알부민 (영양 상태)', 'lab_last_albumin':'최근 알부민',
    'lab_mean_wbc':'백혈구 수 (면역/염증)', 'lab_last_wbc':'최근 백혈구',
    'lab_mean_plt':'혈소판 수', 'lab_min_plt':'혈소판 최저',
    'lab_mean_na':'혈중 나트륨', 'lab_mean_k':'혈중 칼륨',
    'lab_min_k':'혈중 칼륨 최저', 'lab_max_k':'혈중 칼륨 최고',
    'lab_mean_alt':'간수치 ALT', 'lab_mean_ast':'간수치 AST',
    'lab_mean_bun':'혈액요소질소 (신장)', 'lab_last_bun':'최근 혈액요소질소',
    'lab_mean_inr':'혈액응고 INR', 'lab_max_inr':'INR 최고',
    'lab_mean_aptt':'혈액응고 시간 APTT',
    'lab_mean_tbili':'총 빌리루빈 (간)', 'lab_max_tbili':'빌리루빈 최고',
    'lab_mean_crp':'C반응성 단백 (염증)', 'lab_max_crp':'염증 수치 최고',
    'lab_mean_hct':'적혈구 용적률',
    'lab_mean_calcium':'혈중 칼슘',
    'lab_mean_magnesium':'혈중 마그네슘',
    'lab_mean_pt':'프로트롬빈 시간',
    'lab_mean_ldh':'LDH (젖산탈수소효소)',
    'lab_mean_protein':'혈청 총단백',
    'lab_mean_alp':'알칼리성 인산분해효소',
}

UNIT_MAP = {
    'age':'세', 'weight':'kg', 'height':'cm',
    'mean_bt':'°C', 'min_bt':'°C', 'max_bt':'°C', 'last_bt':'°C',
    'mean_hr':'회/분', 'min_hr':'회/분', 'max_hr':'회/분', 'last_hr':'회/분',
    'mean_rr':'회/분', 'mean_spo2':'%', 'min_spo2':'%',
    'mean_nibp_sbp':'mmHg', 'min_nibp_sbp':'mmHg', 'max_nibp_sbp':'mmHg',
    'mean_nibp_dbp':'mmHg', 'mean_nibp_mbp':'mmHg',
    'mean_uo':'mL', 'max_uo':'mL',
    'lab_mean_creatinine':'mg/dL', 'lab_last_creatinine':'mg/dL',
    'lab_mean_glucose':'mg/dL', 'lab_last_glucose':'mg/dL',
    'lab_mean_hb':'g/dL', 'lab_last_hb':'g/dL',
    'lab_mean_albumin':'g/dL', 'lab_last_albumin':'g/dL',
    'lab_mean_na':'mEq/L', 'lab_mean_k':'mEq/L',
    'lab_mean_alt':'IU/L', 'lab_mean_ast':'IU/L',
    'lab_mean_bun':'mg/dL', 'lab_mean_crp':'mg/L',
    'lab_mean_calcium':'mg/dL', 'lab_mean_pt':'초',
    'diag_count':'개',
}

def to_kr(col):
    return FEATURE_KR.get(col, col.replace('_', ' '))

def fmt_val(col, val):
    val = float(val)
    # 이진 피처
    if col in ['emop']: return '응급' if val >= 0.5 else '계획수술'
    if col in ['diag_circulatory','diag_neoplasm','diag_endocrine','diag_respiratory',
               'diag_digestive','diag_musculoskeletal','diag_genitourinary','diag_injury',
               'diag_nervous','diag_mental']:
        return '있음' if val >= 0.5 else '없음'
    unit = UNIT_MAP.get(col, '')
    return f"{val:.1f}{unit}" if val != int(val) else f"{int(val)}{unit}"

# ── 진료과 매핑 ──────────────────────────────────────────────
DEPT_MAP = {
    '일반외과': 'GS', '흉부외과': 'CTS', '신경외과': 'NS',
    '정형외과': 'OS', '산부인과': 'OG', '비뇨기과': 'UR',
    '성형외과': 'PS', '안과': 'OL',
    # 영어도 허용
    'GS':'GS', 'CTS':'CTS', 'NS':'NS', 'OS':'OS',
    'OG':'OG', 'UR':'UR', 'PS':'PS', 'OL':'OL',
}

def safe_encode(col, val):
    le = encoders.get(col)
    if le is None: return medians.get(col, 0)
    try:
        return int(le.transform([str(val)])[0])
    except ValueError:
        return medians.get(col, 0)

@app.route('/surgeries')
def get_surgeries():
    """진료과별 수술 코드 목록 반환"""
    if pcs_lookup is None:
        return jsonify({'error': 'icd10_pcs_lookup.pkl 없음. train.py 재실행 필요'}), 500

    dept = request.args.get('dept', '')   # 예: 'GS'
    dept_map_rev = {v: k for k, v in DEPT_MAP.items() if len(v) <= 3}
    dept_name = dept_map_rev.get(dept, dept)

    if dept_name:
        filtered = pcs_lookup[pcs_lookup['department_name'] == dept_name]
    else:
        filtered = pcs_lookup

    result = []
    for _, row in filtered.iterrows():
        result.append({
            'code':  row['icd10_pcs_code'],
            'label': f"{row['icd10_pcs_code']} (평균 {row['avg_duration']:.0f}분, {int(row['count'])}건)",
            'avg_duration': row['avg_duration'],
            'count': int(row['count']),
        })
    return jsonify(result)


@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

@app.route('/predict', methods=['POST'])
def predict():
    d = request.json

    # 진료과 처리
    dept_raw = d.get('dept', '')
    # "일반외과" or "GS" 둘 다 처리
    dept_code = DEPT_MAP.get(dept_raw, dept_raw)

    # 기본값: 학습 데이터 중앙값
    row = {f: medians.get(f, 0) for f in features}

    # 기본 정보
    row['age']          = float(d.get('age', medians.get('age', 55)))
    row['weight']       = float(d.get('weight', medians.get('weight', 65)))
    row['height']       = float(d.get('height', medians.get('height', 165)))
    row['asa']          = float(d.get('asa', medians.get('asa', 2)))
    row['emop']         = float(d.get('emop', 0))
    row['diag_count']   = float(d.get('diag_count', 0))
    row['surgery_rank'] = 1.0  # 첫 수술 가정
    row['sex']          = safe_encode('sex',        d.get('sex', 'M'))
    row['race']         = safe_encode('race',       'Korean')
    row['department']   = safe_encode('department', dept_code)
    row['antype']       = safe_encode('antype',     d.get('antype', 'General'))
    # icd10_pcs: 프론트에서 실제 코드(예: '0WJG0')를 직접 전달받음
    icd10_pcs_code = d.get('icd10_pcs_code', '')
    row['icd10_pcs'] = safe_encode('icd10_pcs', icd10_pcs_code)

    # 진단명 (체크박스)
    for col in ['diag_circulatory','diag_neoplasm','diag_endocrine','diag_respiratory',
                'diag_digestive','diag_musculoskeletal','diag_genitourinary','diag_injury',
                'diag_nervous','diag_mental']:
        if col in features:
            row[col] = float(d.get(col, 0))

    # 혈액검사 수치 (사용자 입력값 우선, 없으면 중앙값)
    lab_input_map = {
        'lab_creatinine': ['lab_mean_creatinine', 'lab_last_creatinine', 'lab_min_creatinine', 'lab_max_creatinine'],
        'lab_glucose':    ['lab_mean_glucose',    'lab_last_glucose',    'lab_min_glucose',    'lab_max_glucose'],
        'lab_hb':         ['lab_mean_hb',         'lab_last_hb',         'lab_min_hb'],
        'lab_albumin':    ['lab_mean_albumin',     'lab_last_albumin',    'lab_min_albumin',    'lab_max_albumin'],
        'lab_wbc':        ['lab_mean_wbc',         'lab_last_wbc',        'lab_max_wbc'],
        'lab_plt':        ['lab_mean_plt',         'lab_last_plt',        'lab_min_plt'],
        'lab_na':         ['lab_mean_na',          'lab_last_na',         'lab_min_na',         'lab_max_na'],
        'lab_k':          ['lab_mean_k',           'lab_last_k',          'lab_min_k',          'lab_max_k'],
        'lab_alt':        ['lab_mean_alt',         'lab_last_alt'],
        'lab_ast':        ['lab_mean_ast',         'lab_last_ast'],
        'lab_bun':        ['lab_mean_bun',         'lab_last_bun'],
        'lab_inr':        ['lab_mean_inr',         'lab_last_inr',        'lab_max_inr'],
        'lab_crp':        ['lab_mean_crp',         'lab_max_crp'],
        'lab_tbili':      ['lab_mean_tbili',       'lab_max_tbili'],
        'lab_hct':        ['lab_mean_hct'],
        'lab_calcium':    ['lab_mean_calcium'],
        'lab_plt':        ['lab_mean_plt',         'lab_min_plt'],
    }
    for input_key, feat_names in lab_input_map.items():
        val = d.get(input_key)
        if val is not None:
            for fn in feat_names:
                if fn in features:
                    row[fn] = float(val)

    # 예측
    X = pd.DataFrame([row])[features]
    icu_prob    = float(model_cls.predict_proba(X)[0][1]) * 100
    surgery_min = float(model_reg.predict(X)[0])
    surgery_min = max(5.0, min(1440.0, surgery_min))

    # 빠른 경우 / 늦은 경우 (1 표준편차)
    surgery_fast = max(5.0, surgery_min - resid_std * 0.7)
    surgery_slow = min(1440.0, surgery_min + resid_std * 1.2)

    # 중요 피처 추출 (분류 모델)
    imp_cls = model_cls.feature_importances_
    top_cls = np.argsort(imp_cls)[::-1][:8]
    p80 = np.percentile(imp_cls[imp_cls > 0], 80)
    p50 = np.percentile(imp_cls[imp_cls > 0], 50)

    risk_factors = []
    for i in top_cls:
        col = features[i]
        imp = imp_cls[i]
        val = row[col]
        impact = 'high' if imp >= p80 else ('mid' if imp >= p50 else 'ok')
        risk_factors.append({
            'name':   to_kr(col),
            'value':  fmt_val(col, val),
            'impact': impact,
            'raw':    round(float(val), 3),
        })

    # 중요 피처 추출 (회귀 모델)
    imp_reg = model_reg.feature_importances_
    top_reg = np.argsort(imp_reg)[::-1][:8]
    p80r = np.percentile(imp_reg[imp_reg > 0], 80)
    p50r = np.percentile(imp_reg[imp_reg > 0], 50)

    time_factors = []
    for i in top_reg:
        col = features[i]
        imp = imp_reg[i]
        val = row[col]
        impact = 'high' if imp >= p80r else ('mid' if imp >= p50r else 'ok')
        time_factors.append({
            'name':   to_kr(col),
            'value':  fmt_val(col, val),
            'impact': impact,
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
    })

if __name__ == '__main__':
    print("http://localhost:5000 에서 앱 실행 중!")
    app.run(debug=True, port=5000)
