# 실험 목록

| | **실험 제목**                          | **주요 내용**                                                                                 |
|----------|---------------------------------------|--------------------------------------------------------------------------------------------------|
| **1**    | **Salt and Pepper 노이즈 추가**       | 빛 반사, 잉크 번짐 등의 노이즈를 학습시키기 위해 skimage의 random_noise로 Salt and Pepper 노이즈 추가|
| **2**    | **픽셀 Binarization**                  | 픽셀 값을 0 또는 255로 변환하여 텍스트 명확도를 높이기 위해 OpenCV로 Binarization 수행|
| **3**    | **픽셀 Binarization<br>(Adaptive Threshold 적용)**           | 이미지를 영역별로 나눠 다른 임계값을 적용해 OpenCV의 Adaptive Threshold 적용|
| **4**    | **데이터셋에 맞는 Normalize 적용**     | 데이터셋에 맞는 mean, std를 직접 계산하여 Normalize 적용|


<br>
<br>

## 1) Salt and Pepper

### **가정**
- 영수증 데이터는 **빛 반사**, **잉크 번짐** 등의 노이즈가 발생할 수 있으며, 이를 모델에 학습시키면 **강건한 모델**을 만들 수 있을 것이라고 가정했습니다.

---

### **방법**
- **skimage** 라이브러리의 `util.random_noise` 함수를 사용하여 이미지에 **Salt and Pepper 노이즈**를 추가했습니다.

---

### **결론**
- **F1 score가 0.0332 상승**했습니다.

---

### **회고**
- **Salt and Pepper 노이즈**가 영수증 데이터에 대해 **강건한 모델을 만드는 데 도움**을 준다는 것을 확인했습니다.
- 다만, **효과가 제한적**이었으며, 다른 유형의 노이즈(예: Gaussian 노이즈, Motion Blur 등)를 추가해봤다면 더 다양한 실험과 비교가 가능했을 것이라 생각합니다.

---

### **성능 비교**

| **Model**           | **LB (Public)**                                           | **LB (Private)**                                          |
|---------------------|---------------------------------------------------------|---------------------------------------------------------|
| **Base**             | Precision: 0.7176<br>Recall: 0.8512<br>F1_score: **0.7787** | Precision: 0.7129<br>Recall: 0.8429<br>F1_score: **0.7725** |
| **Base + Salt and Pepper** | Precision: 0.7716<br>Recall: 0.8565<br>F1_score: **0.8119** | Precision: 0.7456<br>Recall: 0.8165<br>F1_score: **0.7795** |

<br>
<br>

## 2) Binarization

### **가정**
- 픽셀 값을 **0 또는 255로 변환**하면 텍스트가 더 선명하게 나타나고, 이를 통해 모델이 **더 명확한 특징을 학습**할 수 있을 것이라 가정했습니다.

---

### **방법**
- OpenCV 라이브러리를 이용하고, binarization_threshold는 128로 설정했습니다.

---

### **결론**
- **F1 score가 0.0545 상승**했습니다.

---

### **회고**
- **Binarization**이 영수증 데이터의 **성능 향상에 긍정적인 영향을 미친 것**을 확인했습니다.
- 다만, **binarization_threshold 값을 다양하게 조정**하면서 학습했다면 더 다양한 결과를 얻을 수 있었을 것이라 생각합니다.

---

### **성능 비교**

| **Model**           | **LB (Public)**                                           | **LB (Private)**                                          |
|---------------------|---------------------------------------------------------|---------------------------------------------------------|
| **Base**             | Precision: 0.7176<br>Recall: 0.8512<br>F1_score: **0.7787** | Precision: 0.7129<br>Recall: 0.8429<br>F1_score: **0.7725** |
| **Base + Binarization** | Precision: 0.8596<br>Recall: 0.8083<br>F1_score: **0.8332** | Precision: 0.8300<br>Recall: 0.7915<br>F1_score: **0.8103** |

<br>
<br>

## 3) Binarization (Adaptive Threshold)

### **가정**
- 이미지를 **작은 영역으로 나누고 각 영역마다 다른 임계값을 적용**하는 Adaptive Thresholding 방식을 사용하면, 다양한 명암 조건이 존재하는 이미지에서 **더 효과적**일 것이라고 가정했습니다.

---

### **방법**
- **OpenCV** 라이브러리를 사용하여 **Adaptive Threshold**를 적용했습니다.

---

### **결론**
- **Adaptive Threshold**를 적용한 데이터 셋에 맞는 Normalize 값을 계산해 적용 후 output.csv를 확인해보니, 박스가 하나도 잡히지 않았습니다.
- 이에, Normalize를 기본 설정 0.5로 변경 후 다시 학습 시킨 결과 F1 score가 떨어졌습니다. 

---

### **회고**
- 기본 데이터셋과 **Adaptive Threshold**를 적용한 데이터셋의 **Normalize 계산 값이 크게 차이**가 났습니다.
  - 기본 데이터셋의 Normalize mean: **0.6**
  - Super Resolution + Remove Dash 데이터셋의 Normalize mean: **0.6**
  - Adaptive Threshold를 적용한 데이터셋의 Normalize mean: **0.8**
- 이러한 차이로 인해 **테스트 시 Normalize mean 0.5**와의 차이로 박스가 하나도 잡히지 않았을 가능성이 높다고 생각했습니다.
- **Normalize를 빼고 Adaptive Threshold만 적용**하는 실험도 해보았더라면 더 다양한 분석이 가능했을 것이라 생각합니다.

---

### **성능 비교**

| **Model**                             | **LB (Public)**                                           | **LB (Private)**                                          |
|---------------------------------------|---------------------------------------------------------|---------------------------------------------------------|
| **Base + Normalize (데이터셋에 맞는 값)** | Precision: 0.8248<br>Recall: 0.8604<br>F1_score: **0.8422** | Precision: 0.8047<br>Recall: 0.8252<br>F1_score: **0.8148** |
| **Base + Normalize (0.5) + Binarization** | Precision: 0.7631<br>Recall: 0.7297<br>F1_score: **0.7460** | Precision: 0.7449<br>Recall: 0.7013<br>F1_score: **0.7224** |

<br>
<br>

## 4) Normalize

### **가정**
- 기존에는 **mean=0.5, std=0.5**로 Normalize를 적용했습니다.
- 하지만 사용하는 **이미지 데이터에 맞는 mean, std 값을 직접 계산하여 적용**하면 더 나은 Normalize 효과를 얻을 수 있을 것이라고 가정했습니다.

---

### **방법**
- **ipynb 파일**을 생성하여 **사용 중인 데이터의 mean, std를 직접 계산**했습니다.
- 계산한 mean, std 값을 기존 Normalize 설정(mean=0.5, std=0.5) 대신 **데이터셋에 맞는 값으로 적용**했습니다.

---

### **결론**
- **F1 score가 0.0635 상승**했습니다.

---

### **회고**
- **데이터셋에 맞춘 Normalize 값**을 사용하는 것이 **성능 향상에 긍정적인 영향을 미쳤음**을 확인했습니다.
- 하지만, **테스트 시에는 mean=0.5, std=0.5로 Normalize**한 상태인데, 이 경우에도 **성능이 향상된 이유는 명확히 해석되지 않았습니다**.
- 데이터의 **특성에 맞는 Normalize가 중요한 영향을 미친다는 점**은 확인되었으므로, **테스트 과정에서도 동일한 Normalize 값을 적용하는 실험**이 필요하다고 생각합니다.

---

### **성능 비교**

| **Model**           | **LB (Public)**                                           | **LB (Private)**                                          |
|---------------------|---------------------------------------------------------|---------------------------------------------------------|
| **Base**             | Precision: 0.7176<br>Recall: 0.8512<br>F1_score: **0.7787** | Precision: 0.7129<br>Recall: 0.8429<br>F1_score: **0.7725** |
| **Base + Normalize** | Precision: 0.8248<br>Recall: 0.8604<br>F1_score: **0.8422** | Precision: 0.8047<br>Recall: 0.8252<br>F1_score: **0.8148** |
