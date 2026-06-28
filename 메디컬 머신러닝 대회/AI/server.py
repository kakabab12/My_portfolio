import re
import math
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)
CORS(app)

print("=" * 55)
print("  SurgicalAI 서버 로딩 중...")
print("=" * 55)

model_cls = joblib.load('model_cls.pkl')
model_reg = joblib.load('model_reg.pkl')
encoders  = joblib.load('encoders.pkl')
medians   = joblib.load('medians.pkl')
features  = joblib.load('features.pkl')

try:
    resid_std = float(joblib.load('resid_std.pkl'))
except:
    resid_std = 45.0

try:
    pcs_lookup = joblib.load('icd10_pcs_lookup.pkl')
    print(f"✅ 수술 코드 로드: {len(pcs_lookup)}개")
    print(f"   진료과 목록: {sorted(pcs_lookup['department_name'].unique())}")
except Exception as e:
    pcs_lookup = None
    print(f"⚠ icd10_pcs_lookup.pkl 없음: {e}")

try:
    lab_feature_map = joblib.load('lab_feature_map.pkl')
    print(f"✅ 혈액검사 매핑: {len(lab_feature_map)}개")
except:
    lab_feature_map = {}
    print("⚠ lab_feature_map.pkl 없음")

# 인코더 역변환 (숫자 → 원래 문자열)
inverse_encoders = {}
for col, le in encoders.items():
    inverse_encoders[col] = {i: cls for i, cls in enumerate(le.classes_)}

print(f"✅ 피처: {len(features)}개 | 불확실성: ±{resid_std:.0f}분")

# ── 한글 라벨 ─────────────────────────────────────────────
FEATURE_KR = {
    'age': '나이', 'sex': '성별', 'weight': '체중', 'height': '신장',
    'asa': 'ASA 신체등급', 'emop': '응급수술 여부',
    'department': '진료과', 'antype': '마취 방법',
    'icd10_pcs': '수술 종류 코드', 'diag_count': '기저질환 수',
    'surgery_rank': '수술 순번',
    'diag_circulatory':     '심혈관 질환',
    'diag_neoplasm':        '종양/암',
    'diag_endocrine':       '당뇨/내분비',
    'diag_respiratory':     '호흡기 질환',
    'diag_digestive':       '소화기 질환',
    'diag_musculoskeletal': '근골격계',
    'diag_genitourinary':   '비뇨생식기',
    'diag_injury':          '외상/손상',
    'diag_nervous':         '신경계 질환',
    'diag_mental':          '정신건강',
    'mean_bt': '평균 체온', 'std_bt': '체온 변동성',
    'mean_hr': '평균 심박수', 'std_hr': '심박수 변동성',
    'min_hr': '최저 심박수', 'max_hr': '최고 심박수',
    'mean_rr': '평균 호흡수', 'std_rr': '호흡수 변동성',
    'mean_spo2': '평균 산소포화도', 'min_spo2': '최저 산소포화도',
    'mean_nibp_sbp': '수축기 혈압 (평균)',
    'std_nibp_sbp': '수축기 혈압 변동성',
    'min_nibp_sbp': '수축기 혈압 최저',
    'max_nibp_sbp': '수축기 혈압 최고',
    'mean_nibp_dbp': '이완기 혈압 (평균)',
    'std_nibp_dbp': '이완기 혈압 변동성',
    'mean_nibp_mbp': '평균동맥압',
    'mean_uo': '평균 소변량', 'std_uo': '소변량 변동성',
}

LAB_KR = {
    'lab_creatinine': '크레아티닌 (신장)',
    'lab_glucose':    '혈당',
    'lab_hb':         '혈색소 (빈혈)',
    'lab_albumin':    '알부민 (영양)',
    'lab_wbc':        '백혈구 (염증/면역)',
    'lab_plt':        '혈소판',
    'lab_na':         '혈중 나트륨',
    'lab_k':          '혈중 칼륨',
    'lab_alt':        '간수치 ALT',
    'lab_ast':        '간수치 AST',
    'lab_bun':        '혈액요소질소 (신장)',
    'lab_crp':        'CRP (염증)',
    'lab_inr':        'INR (혈액응고)',
    'lab_tbili':      '총 빌리루빈',
    'lab_hct':        '적혈구 용적률',
    'lab_calcium':    '혈중 칼슘',
    'lab_sbp':        '수축기 혈압',
    'lab_dbp':        '이완기 혈압',
}

LAB_UNITS = {
    'lab_creatinine': 'mg/dL', 'lab_glucose': 'mg/dL',
    'lab_hb': 'g/dL', 'lab_albumin': 'g/dL',
    'lab_wbc': 'x10³/uL', 'lab_plt': 'x10³/uL',
    'lab_na': 'mEq/L', 'lab_k': 'mEq/L',
    'lab_alt': 'IU/L', 'lab_ast': 'IU/L',
    'lab_bun': 'mg/dL', 'lab_crp': 'mg/L',
    'lab_inr': '', 'lab_tbili': 'mg/dL',
    'lab_sbp': 'mmHg', 'lab_dbp': 'mmHg',
}

VITAL_UNITS = {
    'age': '세', 'weight': 'kg', 'height': 'cm',
    'mean_bt': '°C', 'mean_hr': '회/분', 'mean_rr': '회/분',
    'mean_spo2': '%', 'min_spo2': '%',
    'mean_nibp_sbp': 'mmHg', 'min_nibp_sbp': 'mmHg',
    'max_nibp_sbp': 'mmHg', 'mean_nibp_dbp': 'mmHg',
    'mean_nibp_mbp': 'mmHg', 'mean_uo': 'mL', 'diag_count': '개',
}

DEPT_KR = {
    'GS': '일반외과', 'CTS': '흉부외과', 'NS': '신경외과',
    'OS': '정형외과', 'OG': '산부인과', 'UR': '비뇨기과',
    'PS': '성형외과', 'OL': '안과', 'OT': '기타',
}

def feat_to_kr(feat):
    if feat in FEATURE_KR:
        return FEATURE_KR[feat]
    for ik, fl in lab_feature_map.items():
        if feat in fl:
            return LAB_KR.get(ik, feat)
    return feat.replace('_', ' ')

def fmt_val(feat, val):
    val = float(val)
    if feat == 'department':
        orig = inverse_encoders.get('department', {}).get(int(val), str(int(val)))
        return DEPT_KR.get(orig, orig)
    if feat == 'antype':
        orig = inverse_encoders.get('antype', {}).get(int(val), str(int(val)))
        antype_kr = {'General': '전신마취', 'Neuraxial': '척추마취',
                     'Regional': '부위마취', 'MAC': '수면마취'}
        return antype_kr.get(orig, orig)
    if feat == 'icd10_pcs':
        orig = inverse_encoders.get('icd10_pcs', {}).get(int(val), '')
        return orig if orig else '미선택'
    if feat == 'sex':
        orig = inverse_encoders.get('sex', {}).get(int(val), '')
        return '남성' if orig == 'M' else '여성'
    if feat == 'race':
        return ''
    if feat == 'emop':
        return '응급수술' if val >= 0.5 else '계획수술'
    if feat == 'asa':
        return 'ASA ' + str(int(val)) + '등급'
    if feat.startswith('diag_') and feat != 'diag_count':
        return '있음' if val >= 0.5 else '없음'
    for ik, fl in lab_feature_map.items():
        if feat in fl:
            unit = LAB_UNITS.get(ik, '')
            return str(round(val, 1)) + unit
    unit = VITAL_UNITS.get(feat, '')
    return (str(round(val, 1)) if val != int(val) else str(int(val))) + unit

def safe_encode(col, val):
    le = encoders.get(col)
    if le is None:
        return medians.get(col, 0)
    try:
        return int(le.transform([str(val)])[0])
    except:
        return medians.get(col, 0)

def get_asa(d):
    raw = str(d.get('asa', '') or '')
    m = re.search(r'\d+', raw)
    return float(m.group()) if m else float(medians.get('asa', 2))

# ── Evidence-Based 임상 보정 ─────────────────────────────
def clinical_calibrate(base_prob, d, asa_val):
    boost = 0.0

    if asa_val >= 5:   boost += 2.2
    elif asa_val >= 4: boost += 1.1
    elif asa_val >= 3: boost += 0.35

    emop_val = float(d.get('emop', 0) or 0)
    if emop_val >= 1:
        boost += 0.8
        if asa_val >= 4:
            boost += 0.5

    if int(d.get('diag_neoplasm',        0) or 0): boost += 0.4
    if int(d.get('diag_circulatory',     0) or 0): boost += 0.5
    if int(d.get('diag_respiratory',     0) or 0): boost += 0.35
    if int(d.get('diag_endocrine',       0) or 0): boost += 0.2
    if int(d.get('diag_genitourinary',   0) or 0): boost += 0.25
    if int(d.get('diag_nervous',         0) or 0): boost += 0.3

    creat = d.get('lab_creatinine')
    if creat:
        c = float(creat)
        if c > 5.0:   boost += 0.8
        elif c > 2.0: boost += 0.4
        elif c > 1.5: boost += 0.15

    alb = d.get('lab_albumin')
    if alb:
        a = float(alb)
        if a < 2.5:   boost += 0.6
        elif a < 3.0: boost += 0.3
        elif a < 3.5: boost += 0.1

    crp = d.get('lab_crp')
    if crp:
        c = float(crp)
        if c > 100: boost += 0.4
        elif c > 50: boost += 0.2

    hb = d.get('lab_hb')
    if hb:
        h = float(hb)
        if h < 8.0:    boost += 0.4
        elif h < 10.0: boost += 0.2

    wbc = d.get('lab_wbc')
    if wbc:
        w = float(wbc)
        if w > 15 or w < 2: boost += 0.3

    inr = d.get('lab_inr')
    if inr and float(inr) > 2.0:
        boost += 0.3

    sbp = d.get('lab_sbp')
    if sbp:
        s = float(sbp)
        if s > 180:   boost += 0.35
        elif s > 160: boost += 0.15
        elif s < 90:  boost += 0.5

    dbp = d.get('lab_dbp')
    if dbp:
        db = float(dbp)
        if db > 110:  boost += 0.2
        elif db < 60: boost += 0.25

    base_prob = max(0.5, min(99.5, base_prob))
    logit = math.log(base_prob / (100 - base_prob))
    return 100.0 / (1 + math.exp(-(logit + boost)))

# ── /surgeries — 진료과별 엄격 필터링 ───────────────────────
@app.route('/surgeries')
def get_surgeries():
    if pcs_lookup is None:
        return jsonify([])

    dept = request.args.get('dept', '').strip()
    print(f"  /surgeries: dept='{dept}'")
    print(f"  저장된 진료과: {sorted(pcs_lookup['department_name'].unique())}")

    if not dept:
        return jsonify([])

    # 정확히 일치하는 진료과만 (다른 진료과 절대 섞지 않음)
    filtered = pcs_lookup[pcs_lookup['department_name'] == dept].copy()
    print(f"  필터 결과: {len(filtered)}건")

    if filtered.empty:
        print(f"  ⚠ '{dept}' 수술 코드 없음 → generate_lookup.py 재실행 필요")
        return jsonify([])

    filtered = filtered.sort_values('count', ascending=False)

    result = []
    for _, r in filtered.iterrows():
        code = str(r['icd10_pcs_code'])
        desc = ''
        if 'desc' in r.index:
            raw_desc = str(r['desc'])
            if raw_desc not in ('', 'nan', 'None'):
                desc = raw_desc
        avg = int(r['avg_duration'])
        cnt = int(r['count'])

        if desc:
            label = code + '  (' + desc + ')  — 평균 ' + str(avg) + '분 · ' + str(cnt) + '건'
        else:
            label = code + '  — 평균 ' + str(avg) + '분 · ' + str(cnt) + '건'

        result.append({
            'code': code,
            'label': label,
            'avg_duration': float(r['avg_duration']),
        })

    return jsonify(result)

@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

# ── /predict ─────────────────────────────────────────────
@app.route('/predict', methods=['POST'])
def predict():
    d = request.json
    asa_val = get_asa(d)

    row = {f: medians.get(f, 0) for f in features}
    row['age']          = float(d.get('age') or medians.get('age', 55))
    row['weight']       = float(d.get('weight') or medians.get('weight', 65))
    row['height']       = float(d.get('height') or medians.get('height', 165))
    row['asa']          = asa_val
    row['emop']         = float(d.get('emop', 0) or 0)
    row['diag_count']   = float(d.get('diag_count') or 0)
    row['surgery_rank'] = 1.0
    row['sex']          = safe_encode('sex',        d.get('sex', 'M'))
    row['race']         = safe_encode('race',       'Korean')
    row['department']   = safe_encode('department', d.get('dept', ''))
    row['antype']       = safe_encode('antype',     d.get('antype', 'General'))
    row['icd10_pcs']    = safe_encode('icd10_pcs',  d.get('icd10_pcs_code', ''))

    for col in ['diag_circulatory', 'diag_neoplasm', 'diag_endocrine',
                'diag_respiratory', 'diag_digestive', 'diag_musculoskeletal',
                'diag_genitourinary', 'diag_injury', 'diag_nervous', 'diag_mental']:
        if col in features:
            row[col] = float(d.get(col, 0) or 0)

    # 혈액검사 + 혈압 매핑
    lab_feature_map_ext = dict(lab_feature_map)
    lab_feature_map_ext['lab_sbp'] = [f for f in features if 'nibp_sbp' in f]
    lab_feature_map_ext['lab_dbp'] = [f for f in features if 'nibp_dbp' in f]

    lab_inputs_used = []
    for ik in ['lab_creatinine', 'lab_glucose', 'lab_hb', 'lab_albumin',
               'lab_wbc', 'lab_plt', 'lab_na', 'lab_k',
               'lab_alt', 'lab_ast', 'lab_bun', 'lab_crp',
               'lab_inr', 'lab_tbili', 'lab_hct', 'lab_calcium',
               'lab_sbp', 'lab_dbp']:
        val = d.get(ik)
        if val is None or val == '':
            continue
        val = float(val)
        for feat in lab_feature_map_ext.get(ik, []):
            if feat.startswith('diag_'):
                continue
            if feat in features:
                row[feat] = val
                lab_inputs_used.append((ik, feat, val))

    X = pd.DataFrame([row])[features]
    raw_prob    = float(model_cls.predict_proba(X)[0][1]) * 100
    surgery_min = float(model_reg.predict(X)[0])
    surgery_min = max(5.0, min(1440.0, surgery_min))
    icu_prob    = round(clinical_calibrate(raw_prob, d, asa_val), 1)
    surgery_fast = max(5.0, surgery_min - resid_std * 0.7)
    surgery_slow = min(1440.0, surgery_min + resid_std * 1.2)

    feat_to_ik = {}
    for ik, fl in lab_feature_map_ext.items():
        for f in fl:
            feat_to_ik[f] = ik

    user_feats = set(f for _, f, _ in lab_inputs_used)
    for col in ['diag_circulatory', 'diag_neoplasm', 'diag_endocrine',
                'diag_respiratory', 'diag_digestive', 'diag_musculoskeletal',
                'diag_genitourinary', 'diag_injury']:
        if d.get(col, 0):
            user_feats.add(col)

    def get_top_factors(imp, n=7):
        idx = np.argsort(imp)[::-1]
        out, seen_f, seen_ik = [], set(), set()
        # 사용자 입력 피처 우선
        for i in idx:
            f  = features[i]
            ik = feat_to_ik.get(f)
            if f in user_feats:
                if ik and ik in seen_ik:
                    continue
                out.append(i)
                seen_f.add(f)
                if ik:
                    seen_ik.add(ik)
        # 나머지 중요도 순
        for i in idx:
            f  = features[i]
            ik = feat_to_ik.get(f)
            if f in seen_f or f == 'race':
                continue
            if ik and ik in seen_ik:
                continue
            if len(out) >= n:
                break
            out.append(i)
            seen_f.add(f)
            if ik:
                seen_ik.add(ik)
        return out

    imp_cls = model_cls.feature_importances_
    imp_reg = model_reg.feature_importances_
    p80c = np.percentile(imp_cls[imp_cls > 0], 80)
    p50c = np.percentile(imp_cls[imp_cls > 0], 50)
    p80r = np.percentile(imp_reg[imp_reg > 0], 80)
    p50r = np.percentile(imp_reg[imp_reg > 0], 50)

    def make_factors(imp, p80, p50):
        seen_names = set()
        result = []
        for i in get_top_factors(imp):
            name = feat_to_kr(features[i])
            if not name or name in seen_names:
                continue
            seen_names.add(name)
            result.append({
                'name':    name,
                'value':   fmt_val(features[i], row[features[i]]),
                'impact':  'high' if imp[i] >= p80 else ('mid' if imp[i] >= p50 else 'ok'),
                'is_user': features[i] in user_feats,
            })
        return result

    risk_level = '고위험' if icu_prob >= 50 else ('중위험' if icu_prob >= 20 else '저위험')
    rec = (
        '해당 환자는 ICU 입원 확률 ' + str(icu_prob) + '%로 ' + risk_level +
        '군으로 분류됩니다. 예상 수술 시간은 약 ' + str(round(surgery_min)) +
        '분이며 빠르면 ' + str(round(surgery_fast)) +
        '분, 늦으면 ' + str(round(surgery_slow)) + '분 예상됩니다. ' +
        ('▶ 수술 전 ICU 사전 배정 및 집중치료팀 준비를 강력 권고합니다.'
         if icu_prob >= 50 else
         '▶ 표준 수술 프로토콜 적용, 활력징후 모니터링을 권고합니다.'
         if icu_prob >= 20 else
         '▶ 일반 회복실 배정 후 표준 경과 관찰이 적합합니다.')
    )

    return jsonify({
        'icu_prob':       icu_prob,
        'raw_prob':       round(raw_prob, 1),
        'surgery_min':    round(surgery_min),
        'surgery_fast':   round(surgery_fast),
        'surgery_slow':   round(surgery_slow),
        'risk_factors':   make_factors(imp_cls, p80c, p50c),
        'time_factors':   make_factors(imp_reg, p80r, p50r),
        'recommendation': rec,
        'lab_count':      len(lab_inputs_used),
        'risk_level':     risk_level,
    })

if __name__ == '__main__':
    print("http://localhost:5000")
    app.run(debug=True, port=5000)
