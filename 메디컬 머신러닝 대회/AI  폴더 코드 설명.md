1. AI 모델 학습 및 전처리 (Model Training & Pipeline)
## train.py (메인 학습 파이프라인)

역할: 원본 의료 데이터(operations, diagnosis, labs, ward_vitals.csv)를 병합하고 전처리하여 최종 머신러닝 모델을 학습시키는 핵심 스크립트입니다.

특징: 수술 전 데이터만 사용하도록 시간(cutoff_time)을 엄격하게 통제하여 '미래 데이터 참조(Data Leakage)'를 방지했습니다. LightGBM 알고리즘을 사용해 ICU 입원을 예측하는 분류 모델(Classifier)과 수술 시간을 예측하는 회귀 모델(Regressor) 두 가지를 동시에 학습시키고 .pkl 파일로 저장합니다.

## test.py (평가 및 SHAP 추출)

역할: 데이터 전처리 및 모델 학습 후, SHAP(설명 가능한 AI) 라이브러리를 활용해 변수 중요도를 시각화하는 평가 스크립트입니다.

특징: 분류 모델과 회귀 모델 각각에 대해 어떤 요인(나이, 심박수, 혈압 등)이 예측에 가장 큰 영향을 미쳤는지 Summary Plot 이미지(shap_classification.png 등)로 뽑아냅니다.

 2. XAI 및 성능 시각화 (Visualization)
## visualize_metrics.py (성능 지표 시각화)

역할: 모델의 평가 지표(AUROC, Accuracy, Precision, F1-Score 등)를 논문이나 발표 자료에 쓰기 좋게 깔끔한 막대그래프로 그려주는 스크립트입니다.

특징: 운영체제(Windows, Mac)를 자동 감지하여 한글 폰트 깨짐 현상을 방지하는 디테일이 들어있습니다.

## visualize_regression.py (특정 환자 맞춤형 XAI 시각화)

역할: "이 환자는 왜 수술 시간이 오래 걸릴까?"를 의료진에게 설명하기 위해 SHAP 폭포수(Waterfall) 차트를 생성합니다.

특징: 고령, ASA 4등급 등 특정 가상 환자의 데이터를 입력하면, 해당 환자의 수술 시간을 늘리거나 줄인 결정적 요인들을 시각적으로 분석해 줍니다.

 3. 백엔드 API 서버 (Backend Server)
## server.py (LightGBM 기반 메인 서버)

역할: 프론트엔드(웹)에서 환자 정보를 받아 AI 모델로 추론한 뒤 결과를 반환해 주는 Flask 기반 API 서버입니다.

특징: AI가 내놓은 단순 확률값에 의료 지식(ASA 등급, 응급 여부, 혈액검사 수치 등)을 더해 위험도를 보정하는 임상 보정(Evidence-Based Calibration) 로직이 적용되어 있어 실제 의료 현장에 적합하도록 설계되었습니다.

## serverbro.py (XGBoost 통합 업그레이드 서버)

역할: server.py의 업그레이드 버전입니다.

특징: 수술 시간 예측은 기존 LightGBM을 쓰되, ICU 위험도 분류 모델을 더 고도화된 XGBoost 모델(xgboost_high_risk_model...json)로 교체하여 하이브리드(앙상블) 형태로 서비스를 구동하도록 만들어졌습니다.

 4. 프론트엔드 웹 대시보드 (Frontend UI)
## index.html (의료진용 스마트 대시보드)

역할: 의사나 간호사가 환자의 나이, 질환, 혈액검사 수치 등을 쉽게 입력하고 예측 결과를 직관적으로 볼 수 있는 UI 화면입니다.

특징: * 별도의 이미지 파일 없이 CSS와 SVG만으로 원형 게이지(위험도)와 애니메이션 효과를 매우 가볍고 세련되게 구현했습니다.

단순히 '위험함/안전함'만 알려주는 것이 아니라, 최단/최장 수술 시간, 그리고 AI가 이 결과를 도출한 '결정적 요인(막대 바 형태)'과 '종합 임상 소견'을 텍스트로 자세히 브리핑해 줍니다.