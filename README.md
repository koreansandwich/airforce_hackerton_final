# 🛫 제7회 공군 AI 해커톤 본선 - 최우수상(2등)

본 리포지토리는 2025년 **공군 AI 해커톤 본선**에서 수행한 두 과제(Task 1, 2)에 대한 문제 정의, 데이터 분석, 모델 설계, 결과 및 회고를 정리한 문서입니다.  
해당 프로젝트는 폐쇄된 군사 인트라넷 환경에서 진행되어, **코드 전체는 보안상 외부 반출이 불가**하며, 이에 따라 **핵심 로직과 구조 설명 중심**으로 구성하였습니다.

---

## 🧠 Overview

| 항목 | 내용 |
|------|------|
| 주최 | 대한민국 공군 |
| 개발 환경 | Docker 기반 Jupyter Notebook<br>RTX 4070 Ti / i7 / RAM 64GB (제한: 40GB) |
| 팀 인원 | 2명 (이태민, 전다함) |
| 수행 과제 | Task 1: 유사 일기도 탐색<br>Task 2: 시정 예측 |
| 수상 | 🥈 최우수상 (2등) |
| 평가 비율 | Task 1 (20%) + Task 2 (20%) + 발표 심사 (60%) |

---

## 🧰 Tech Stack

- **Language & Framework**  
  ![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)  
  ![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)  
  ![AutoGluon](https://img.shields.io/badge/AutoGluon-0099CC?style=for-the-badge&logo=amazonaws&logoColor=white)  
  ![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white)

- **Computer Vision & Data**  
  ![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)  
  ![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)  
  ![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)  
  ![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C?style=for-the-badge&logo=plotly&logoColor=white)

- **GPU 환경**  
  ![CUDA](https://img.shields.io/badge/CUDA-76B900?style=for-the-badge&logo=nvidia&logoColor=white)

---

## 🌍 Task 1: 유사 일기도 탐색 기법 개발

### 📌 목표
주어진 특정 시점의 일기도와 **유사한 기상 패턴**(강수 여부, 등압선 구조, H/L 위치)을 가진 일기도 5장을 순위별로 탐색하는 문제.  
예보관의 **정성 평가**를 통해 유사도를 채점하며, 강수 여부를 맞추는 것도 중요 요소 중 하나.

### 🗂 데이터 구성
- \dataset\WMap: 10년간의 지상 일기도 이미지 (gif 형식, 총 37,393장)
- \dataset\WMap_ref: 240608 09시 기준, 예보관이 유사하다고 판단한 참조 일기도 제공

### 🔨 최종 사용 기법

#### ✅ 1. 전처리
- **색상 마스킹**: 일기도 배경 제거 → 빨간색(H), 파란색(L), 등압선만 추출
- **영역 crop**: 한반도 주변 (100E ~ 140E, 20N ~ 55N) 포함하는 사각형 crop
- **rain flag 계산**: 초록색 픽셀 탐색 → 강수 여부 1/0로 이진화 (한반도 주변만)

#### ✅ 2. Feature 기반 유사도 계산
- **H/L 거리 기반**
  - OpenCV `matchTemplate`로 H/L 기호 위치 탐지
  - 중심 좌표 간 최소 거리 매칭 후 평균 → 유사도 계산
  - L(저기압)에 가중치 부여 (기상 영향 더 큼)

- **등압선 벡터 기반**
  - 등압선만 남긴 이미지에서 H/L 위치 기준 Gaussian Blur 적용
  - BFS로 등압선 단위 넘어가며 **기압 분포 시각화**
  - 밝기 = 기압 → ResNet 기반 Contrastive Learning 임베딩 학습
  - 유사한 분포는 가까운 임베딩 → cosine similarity 계산

- **최종 유사도 산출 방식**
  ```math
  최종 유사도 = (거리 기반 유사도 × 2) + (등압선 기반 유사도 × 1)

#### ✅ 3. Contrastive Learning
- Positive pair: 24시간 이내 일기도
- Negative pair: 그 외 대부분
- 학습 안정화를 위해 positive 비율 조절
- 모델: ResNet 기반 encoder

### 🔍 평가 포인트

- 기상 도메인 지식 없이도 정량화된 요소 중심으로 유사도 정립
- 일기도 해석과 전처리, 시각화, 딥러닝 임베딩까지 멀티모달 융합
- 발표 심사에서 가장 긍정적 평가를 받은 파트 중 하나

## 🌫 Task 2: 시정 예측 기법 개발

### 📌 목표
10일 간의 기상 관측 데이터 → 다음날 3가지 예측값 도출

1. min_visibility: 다음날 최저 시정
2. recovery_16m: 1마일 이상 시정 첫 회복 시간 (단위: 시간)
3. recovery_48m: 3마일 이상 시정 첫 회복 시간

### 🗂 데이터 구성
- \dataset\ObsData: 2014~2022 기상 관측치 (실수형 다변량 시계열)
- \dataset\ObsData_LB: Public 평가용 20건 (연도 랜덤 치환)

### 🔨 최종 사용 기법

#### ✅ 1. 전처리 & 특징 추출
- 날짜/풍향: sin/cos 변환 (주기성 보존)
- categorical 운량 정보: ordinal 값으로 수치화
- 결측치: 선형보간 또는 0 대체
- Autogluon용 데이터는 1일 단위로 압축 (평균, 표준편차 등)

#### ✅ 2. 모델 구조 설계
사용 모델
- AutoGluon TabularPredictor (주력)
- AutoGluon TimeseriesPredictor
- PyTorch LSTM (시도했으나 정제 어려움)

#### ✅ 3. 시도한 전략
- label2/3의 -1 비중이 높음 → 먼저 이진 분류 후 회귀 적용
- min_visibility → 이전날 대비 상대 차이 예측
- 하루 24시간 시정 예측 후 후처리로 회복 시간 계산 (alternative)
- t-SNE로 Public 데이터 분포 분석 → 시정 나쁨 날에 편중됨 확인

#### ✅ 4. 최종 모델 구조

[7일 입력] → [Sunny/Cloudy 이진 분류] → 각각 별도 앙상블 회귀 모델로 3가지 target 예측
                                            └─ label2/3는 이진 분류 → 회귀 조합
#### ✅ 5. 평가 전략

- t-SNE 분석 통해 Public 데이터 분포 차이 인식
- Cloudy 분기 모델에 더 집중
- threshold 조정으로 이진 분류 bias 조절

### 🏁 결과

- Public 기준 리더보드 1위 달성 (t-SNE feature 포함)
- 제출 제한 시간 문제로 일반화 모델 제출 → 최종 2위

### 🧩 코드 전체를 업로드할 수 없는 이유
본 프로젝트는 공군 인트라넷 보안 환경 내에서 수행되었으며,
외부 네트워크 차단, USB 사용 제한, 파일 반출 금지 등 보안 상의 이유로 인해
전체 코드 및 데이터는 외부 저장 불가였습니다.

따라서 이 리포지토리는:

- 핵심 알고리즘 로직과 흐름
- 데이터 전처리 및 구조화 방식
- 모델링 전략과 평가 로직

위 중심으로 기술하였습니다.

### 🧭 회고

- 기상 도메인 없이 AI 기법으로 문제 해결 구조를 수립한 경험
- 단순 딥러닝 모델보다 전처리 및 문제 재정의의 중요성 체감
- 실무/실전형 문제에 대한 팀 기반 협업 역량 배양
- 제약된 환경 내에서 가장 최적의 접근 방식을 설계했던 것이 핵심

