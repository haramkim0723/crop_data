# Sugar Beet YOLO 실험 로그

## 프로젝트 개요
- **목적**: 드론 촬영 이미지에서 작물(class 0)과 잡초(class 1) 탐지
- **모델**: YOLOv12m 기반
- **데이터**: train ~9,772장 / val 2,443장 / test 493장
- **이미지 크기**: 1280x1280

---

## 실험 결과 요약

| 모델 | Precision | Recall | mAP50 | mAP50-95 | 비고 |
|------|-----------|--------|-------|----------|------|
| yolo11m | 0.820 | 0.776 | 0.843 | 0.657 | 베이스라인 |
| yolo26m | 0.812 | 0.764 | 0.831 | 0.658 | 대형 모델, 성능 오히려 낮음 |
| yolo12m | 0.823 | 0.776 | 0.841 | 0.658 | 베이스라인 |
| yolo11m_v2 | 0.810 | 0.773 | 0.841 | 0.656 | - |
| yolo12m_v2 | 0.817 | 0.777 | 0.842 | 0.658 | 잡초 증강 → 효과 없음 |
| yolo12m_v3 | 예정 | 예정 | 예정 | 예정 | 작물 증강 + hard negative |

### 클래스별 성능 (yolo12m_v2 기준)
| 클래스 | Precision | Recall | mAP50 | mAP50-95 |
|--------|-----------|--------|-------|----------|
| 작물 | 0.781 | 0.670 | 0.758 | 0.497 |
| 잡초 | 0.855 | 0.881 | 0.931 | 0.824 |

---

## 시행착오 기록

### 1. yolo12m_v2 — 잡초 증강 (실패)

**변경 사항**
- `augment_weed.py` 실행: 잡초 전용 이미지 플립/회전 → 약 12,196장 추가
- `cls=1.0` (기본 0.5에서 2배)
- `copy_paste=0.3`
- `close_mosaic=20` (기본 10에서 증가)

**결과**: 성능 변화 없음 (mAP50: 0.841 → 0.842)

**원인 분석**
- 잡초는 이미 mAP50 0.931로 충분히 잘 됨
- 약점은 **작물(mAP50 0.758)**인데 엉뚱한 클래스를 증강
- `close_mosaic=20`은 fine-tuning 구간을 줄여 역효과 가능성
- `cls=1.0`도 성능 개선 없음

**교훈**: 증강 전 클래스별 성능 확인 필수. 잘 되는 클래스 증강은 의미 없음.

---

### 2. confusion matrix 분석 — 진단 수정

**초기 가설 (잘못된 진단)**
> crop ↔ weed 혼동이 문제일 것

**실제 confusion matrix 결과**

| 오류 유형 | 비율 |
|-----------|------|
| 작물 → 잡초 오분류 | 0.02 (2%) |
| 잡초 → 작물 오분류 | 0.01 (1%) |
| 작물 → background (FN) | **0.28 (28%)** |
| 잡초 → background (FN) | 0.09 (9%) |
| background → 작물 (FP) | **0.77** |
| background → 잡초 (FP) | 0.23 |

**수정된 진단**
- crop ↔ weed 혼동은 1~2%에 불과 → 핵심 문제 아님
- **진짜 문제: crop vs background**
  - 작물 FN 28%: 실제 작물을 배경으로 놓침
  - background → crop FP 높음: 배경(흙/그림자)을 작물로 오탐

**교훈**: mAP 평균만 보지 말고 confusion matrix로 오류 유형을 먼저 확인해야 함.

---

### 3. hard negative 수집 방식 수정 (3단계)

**1차 방식 (문제 있음)**
- FP가 있는 이미지 전체를 train에 복사 + 원본 GT 라벨 유지
- 문제: 잡초가 포함된 이미지도 수집되고, 잡초→작물 혼동 케이스도 섞임

**2차 방식 (문제 있음)**
- FP 박스 영역만 패치로 크롭 + 빈 라벨
- 문제: IoU 0.3 기준이 너무 엄격 → GT 객체가 있는 영역도 FP로 판정됨
- 실제로 잡초가 보이는 패치가 수집되는 현상 발생

**3차 방식 (현재)**
- GT 박스와 조금이라도 겹치면(intersection > 0) 제외
- FP 박스를 GT 박스에 닿기 직전까지 사방으로 최대 확장 후 크롭
- 순수 배경만 포함된 최대 크기 패치 수집

**교훈**: IoU threshold보다 intersection 여부로 판단하는 게 더 안전. 패치는 작게보다 크게.

---

## v3 실험 계획

**데이터 구성**
| 구분 | 수량 |
|------|------|
| 원본 train | ~9,772장 |
| cropaug (작물 전용 플립/회전) | 1,832장 |
| weedaug (절반 삭제 후) | ~6,098장 |
| hard negative 패치 | 수집 중 |

**하이퍼파라미터**
- `cls=0.5` (기본값 복원)
- `copy_paste=0.0` (제거)
- `close_mosaic=10` (기본값 복원)
- `cos_lr` → 우선순위 낮음, 나중에 격리 실험

**성공 기준**
| 지표 | 현재 | 목표 |
|------|------|------|
| 작물 mAP50 | 0.758 | 0.800+ |
| 작물 Recall | 0.670 | 0.750+ |
| 작물 FN | 28% | 20% 이하 |
| background→작물 FP | 0.77 | 감소 또는 유지 |
| 잡초 mAP50 | 0.931 | 유지 |

**실행 순서**
```bash
python collect_hard_negatives.py --cleanup  # 기존 hard negative 삭제
python collect_hard_negatives.py            # FP 패치 재수집 (최대 확장 방식)
python reduce_weedaug.py                    # weedaug 절반 삭제
rm train/labels.cache                       # 캐시 삭제 필수
MLFLOW_TRACKING_URI=http://127.0.0.1:5000 python train_yolo12m_v3.py
```

---

## 학습 중 병행 작업 (v3 돌아가는 동안)

> AI 개발은 학습 중에도 다음 학습을 위한 오류 분석이 본업

### 로그 모니터링 체크리스트
- [ ] train/val loss 같이 내려가는지
- [ ] crop recall이 오르는지 (핵심 지표)
- [ ] early stopping 걸릴 조짐 없는지
- [ ] GPU 메모리/속도 이상 없는지

### 오류 샘플 수집 스크립트
```bash
# crop FN 이미지 200장 추출 (초록=GT놓친것, 빨강=모델예측)
python collect_fn_samples.py

# background→crop FP 패치 수집
python collect_hard_negatives.py
```

### 수집 후 사람이 직접 확인할 것
1. **crop FN** (`fn_samples/` 폴더) — 작물을 놓치는 패턴이 있는가?
   - 작은 작물인가?
   - 흙과 색이 비슷한가?
   - 그림자/흐림 때문인가?
   - 라벨 누락인가?
2. **background→crop FP** — 배경의 어떤 특징을 작물로 착각하는가?
   - 흙 질감?
   - 잔여 식물?
   - 그림자 패턴?

### 다음 실험 설계 기준
| 실험 | 내용 | 조건 |
|------|------|------|
| v3a 결과 분석 | crop FN/FP 변화 확인 | v3 완료 후 |
| mixed 이미지 증강 | 작물+잡초 혼재 이미지 보강 | FN 패턴 확인 후 |
| hard negative 비율 조정 | 현재 비율 효과 없으면 10%로 | FP 변화 보고 |
| cos_lr 격리 실험 | 데이터 고정 후 cos_lr만 변경 | 데이터 전략 확정 후 |

---

## 파일 목록

| 파일 | 역할 |
|------|------|
| `augment_weed.py` | 잡초 전용 이미지 증강 (v2에서 사용, 효과 없었음) |
| `augment_crop.py` | 작물 전용 이미지 증강 |
| `reduce_weedaug.py` | weedaug 파일 절반 랜덤 삭제 |
| `collect_hard_negatives.py` | background→crop FP 패치 수집 (GT 제외 최대 확장) |
| `collect_fn_samples.py` | crop FN 이미지 시각화 수집 (초록=GT, 빨강=예측) |
| `train_yolo12m_v3.py` | v3 학습 스크립트 |
| `upload_to_mlflow.py` | 기존 실험 결과 MLflow 업로드 |
| `run_with_mlflow.sh` | MLflow 서버 + 학습 자동 실행 |
| `compare_models.py` | 모델 간 성능 비교 그래프 |
