import pandas as pd
import joblib
import warnings
warnings.filterwarnings('ignore')

print("수술 코드 목록 생성 중...")
ops = pd.read_csv('operations.csv')
ops['surgery_duration'] = ops['opend_time'] - ops['opstart_time']
ops = ops[(ops['surgery_duration'] >= 5) & (ops['surgery_duration'] <= 1440)]

print(f"전체 수술 건수: {len(ops):,}")
print(f"진료과 목록: {sorted(ops['department'].unique())}")
print(f"상위 수술 코드: {ops['icd10_pcs'].value_counts().head(10).index.tolist()}")

# ── ICD-10-PCS 진료과별 수술명 딕셔너리 ────────────────────
ICD_DESC = {
    # 일반외과 (GS)
    '0FB4': '담낭 절제술 (복강경)', '0FC4': '담낭 절개절제술 (복강경)',
    '0FT4': '담낭 전절제술',
    '0DBG': 'S상결장 절제술', '0DTG': 'S상결장 전절제술',
    '0DBH': '직장 절제술', '0DTH': '직장 전절제술',
    '0DBE': '우측대장 절제술', '0DBF': '좌측대장 절제술',
    '0DTE': '대장 전절제술',
    '0DBN': '소장 절제술', '0DQH': '직장 복구술',
    '0DQG': 'S상결장 복구술', '0DQN': '소장 복구술',
    '0DB':  '위 절제술', '0DT': '위 절단술',
    '0D16': '위-공장 우회술', '0D1B': '위공장문합술',
    '0D1':  '위 우회술', '0DQ': '장 복구술', '0DN': '장 이완술',
    '0FB':  '간 절제술', '0FT': '간 전절제술', '0FQ': '간 복구술',
    '0FBG': '췌장 절제술', '0FTG': '췌장 전절제술', '0FG': '췌장 우회술',
    '0WQF': '서혜부 탈장 복구술', '0WQG': '대퇴 탈장 복구술',
    '0WQ':  '복벽 복구술', '0WU': '복벽 보강술 (메쉬)',
    '0WJ':  '복강 탐색술 (개복)', '0WB': '복강 절제술',
    '0HTT': '유방 전절제술', '0HTU': '유방 부분절제술',
    '0HBT': '유방 절개절제술', '0HT': '유방 절단술',
    '0GB':  '갑상선 전절제술', '0G5': '갑상선 부분절제술',
    '0GT':  '갑상선 절단술',
    # 흉부외과 (CTS)
    '021':    'CABG (관상동맥 우회술)',
    '021009': 'CABG — 내유동맥 사용',
    '0210093':'CABG — 좌내유-전하행지',
    '02H':  '페이스메이커 삽입술',
    '02R':  '심장판막 치환술',
    '02U':  '심장판막 보강술 (성형술)',
    '02Q':  '심장 복구술',
    '027':  '관상동맥 확장술 (스텐트)',
    '02N':  '심낭 이완술',
    '02V':  '심장 제한술',
    '02B':  '심장 절제술',
    '02C':  '심장 절개제거술',
    '0BB':  '폐 절제술 (폐엽)',
    '0BT':  '폐 전절제술',
    '0BB0': '좌폐엽 절제술', '0BB1': '우폐엽 절제술',
    '0BT0': '좌전폐 절제술', '0BT1': '우전폐 절제술',
    '0BC':  '폐 절개제거술',
    '0BQ':  '폐 복구술',
    '0BN':  '흉막 이완술',
    '0BW':  '흉강 수정술',
    '0BH':  '흉강 삽입술 (배액관)',
    '04R':  '대동맥 치환술',
    '04B':  '하지혈관 절제술',
    '04Q':  '하지혈관 복구술',
    '04L':  '혈관 폐색술',
    '047':  '혈관 확장술',
    '04U':  '혈관 보강술',
    '04S':  '혈관 재위치술',
    # 신경외과 (NS)
    '00B':  '뇌 절제술',
    '00C':  '뇌 절개제거술 (혈종제거)',
    '001':  '뇌척수액 단락술 (VP Shunt)',
    '009':  '뇌 배출술',
    '00H':  '두개강내 삽입술',
    '00N':  '신경 이완술',
    '00Q':  '신경 복구술',
    '00W':  '두개강 수정술',
    '0SG0': '경추 융합술',
    '0SG7': '요추 융합술',
    '0SB3': '경추 절제술 (추간판제거)',
    '0SB4': '흉추 절제술',
    '0SB':  '척추 절제술',
    '0SG':  '척추 융합술',
    '0RG0': '경추 후방 융합술',
    # 정형외과 (OS)
    '0SRB': '고관절 전치환술 (THA)',
    '0SRC': '무릎 전치환술 (TKA)',
    '0SR90':'고관절 반치환술',
    '0SR':  '관절 치환술',
    '0RR':  '상지관절 치환술',
    '0QR':  '하지관절 치환술',
    '0RG':  '상지관절 융합술',
    '0QG':  '하지관절 융합술',
    '0RB':  '상지 절제술',
    '0QB':  '하지 절제술',
    '0RQ':  '상지 복구술',
    '0QQ':  '하지 복구술',
    '0RS':  '상지 재위치술',
    '0QS':  '하지 재위치술',
    '0RN':  '상지 이완술 (근막절개)',
    '0QN':  '하지 이완술 (근막절개)',
    '0QT':  '하지 절단술',
    '0RT':  '상지 절단술',
    # 산부인과 (OG)
    '0UT':    '자궁 전절제술',
    '0UT9':   '자궁 전절제술 (복강경)',
    '0UB':    '자궁 부분절제술',
    '0UBC':   '자궁경부 절제술 (LEEP/CKC)',
    '0U5':    '난소 절제술',
    '0UB0':   '좌측난소 절제술',
    '0UB1':   '우측난소 절제술',
    '0UQ':    '자궁 복구술 (근종절제술)',
    '0US':    '자궁 재위치술',
    '0UN':    '자궁 이완술',
    '10D00Z0':'제왕절개술 — 저위횡절개',
    '10D00Z1':'제왕절개술 — 고위종절개',
    '10D':    '제왕절개술',
    '10E0XZZ':'자연분만',
    '0UH':    '자궁내 삽입술',
    # 비뇨기과 (UR)
    '0TB':  '신장 절제술',
    '0TT':  '신장 전절제술',
    '0TB0': '좌측신장 절제술',
    '0TB1': '우측신장 절제술',
    '0TC':  '신장 절개제거술 (결석제거)',
    '0TBB': '방광 절제술',
    '0TTB': '방광 전절제술',
    '0TQ':  '방광 복구술',
    '0TV':  '방광 제한술 (요실금)',
    '0TN':  '방광 이완술',
    '0TH':  '비뇨기 삽입술 (스텐트/카테터)',
    '0T7':  '요관 확장술',
    '0T9':  '방광 배출술',
    '0VT':  '전립선 전절제술 (RP)',
    '0VB':  '전립선 절제술 (TURP)',
    '0VH':  '전립선 삽입술',
    # 성형외과 (PS)
    '0HQ':  '피부 복구술 (봉합/성형)',
    '0HB':  '피부 절제술',
    '0HU':  '피부 보강술 (피부이식)',
    '0HR':  '피부 치환술',
    '0HS':  '피부 재위치술 (피판술)',
    '0HN':  '피부 이완술',
    '0JQ':  '피하조직 복구술',
    '0JU':  '피하조직 보강술 (보형물)',
    '0JR':  '피하조직 치환술',
    # 안과 (OL)
    '08R':  '안구 치환술 (인공수정체)',
    '08Q':  '안구 복구술',
    '08B':  '안구 절제술',
    '08T':  '안구 절단술',
    '08C':  '각막이식술',
    '087':  '눈 확장술',
    '08H':  '안구 삽입술',
    '08N':  '안구 이완술',
    '08V':  '안구 제한술 (사시교정)',
    '08S':  '안구 재위치술',
    # 방사선/기타 (RAD/OT)
    '3E0':  '약물 투여', '4A0': '생체계측 모니터링',
}

def get_desc(code):
    if not code or str(code).strip() in ('nan', ''):
        return ''
    code = str(code).strip()
    if code in ICD_DESC:
        return ICD_DESC[code]
    for length in [6, 5, 4, 3]:
        if len(code) >= length and code[:length] in ICD_DESC:
            return ICD_DESC[code[:length]]
    return ''

# ── 진료과별 집계 ────────────────────────────────────────────
lookup = (
    ops.groupby(['department', 'icd10_pcs'])
    .agg(count=('subject_id', 'count'), avg_duration=('surgery_duration', 'mean'))
    .reset_index()
    .sort_values('count', ascending=False)
)
lookup.columns = ['department_name', 'icd10_pcs_code', 'count', 'avg_duration']
lookup['avg_duration'] = lookup['avg_duration'].round(1)
lookup['desc'] = lookup['icd10_pcs_code'].apply(get_desc)

# ── 핵심: 진료과별 엄격 분리 후 상위 30개 ──────────────────
result_list = []
for dept_name, group in lookup.groupby('department_name'):
    top30 = group.nlargest(30, 'count').copy()
    top30['department_name'] = dept_name  # 진료과 명시적 재지정
    result_list.append(top30)

top = pd.concat(result_list, ignore_index=True)

joblib.dump(top, 'icd10_pcs_lookup.pkl')

print(f"\n✅ 저장 완료!")
print(f"\n진료과별 수술 코드 수:")
for dept, grp in top.groupby('department_name'):
    mapped = grp[grp['desc'] != '']
    print(f"  {dept}: 총 {len(grp)}개 (한글명 {len(mapped)}개)")

print(f"\n샘플 (상위 10개):")
print(top[['department_name','icd10_pcs_code','desc','count','avg_duration']].head(10).to_string(index=False))
