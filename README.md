# BoostCamp AI Tech 7th CV-01 Object Detection Project

## Team Members
| <img src="https://github.com/user-attachments/assets/539241ed-cfcd-4055-8c9f-0766df482995" width="100"/> | <img src="https://github.com/user-attachments/assets/539241ed-cfcd-4055-8c9f-0766df482995
" width="100"/> | <img src="https://github.com/user-attachments/assets/539241ed-cfcd-4055-8c9f-0766df482995
" width="100"/> | <img src="https://github.com/user-attachments/assets/539241ed-cfcd-4055-8c9f-0766df482995
" width="100"> | <img src="https://github.com/user-attachments/assets/539241ed-cfcd-4055-8c9f-0766df482995
" width="100"/> | <img src="https://github.com/user-attachments/assets/539241ed-cfcd-4055-8c9f-0766df482995
" wdith="100/"> |
|:----------------------------------------------------------:|:----------------------------------------------------------:|:----------------------------------------------------------:|:----------------------------------------------------------:|:----------------------------------------------------------:|:----------------------------------------------------------:|
| **동준**                                                     | **경윤**                                                      | **영석**                                                      | **태영**                                                      | **태성**                                                      | **세린**                                                      |

## Contribute

| Member | Roles |
|--------|-------|
| **동준** | K-fold, Dataset Add, Remove dash, Ensemble |
| **영석** | EDA, Streamlit으로 결과 시각화, cv2에서 사용 가능한 이미지 처리 기법 탐색 |
| **경윤** | EDA, 라이브러리 구조파악, Model 선택, 가설 설정 및 실험 |
| **태영** | EDA, super resolution, Ensemble |
| **세린** | EDA, Detectron2 k-fold |
| **태성** | 템플릿 코드 작성, Dataset Add  |


## Overview

When performing deep learning tasks, there are typically two main approaches: one focuses on the model, and the other on the data. In this project, we adopted a data-centric approach to tackle an OCR task related to receipts.

## Dataset

This project utilizes a dataset specifically designed for OCR tasks involving receipts. The dataset contains labeled images of various receipt elements, divided into training and test sets

- **Training Images**: 400
- **Training bboxes**: 34623
- **Test Images**: 120
- **Lanugages**: 4
  - **Chinese**: 100
  - **Japanese**: 100
  - **thai**: 100
  - **Vietnamese**: 100

## Development Environment

| **Category**       | **Details**                        | **Category**       | **Details**            |
|--------------------|------------------------------------|--------------------|------------------------|
| **Hardware**       | GPU: V100 32GB × 4                | **Python**         | 3.10                   |
| **CUDA**           | 12.1                              | **PyTorch**        | 2.1.0                   |
| **PyTorch Lightning** | 1.8.0                           | **Libraries**      |Opencv-python(4.10.0.84), numpy(1.24.4) |
| **Collaboration Tools** | Notion, WandB               |                   |                        |


## Results

### 최종 결과 

<div style="display: flex; flex-direction: column; align-items: center;">
    <div style="text-align: center; margin-bottom: 10px;">
        <img src="https://github.com/user-attachments/assets/244d276e-1e85-4752-984a-680a24193b1f" width="500" />
    </div>
</div>


### Data augmentation

<div>
    <div style="text-align: center; margin-bottom: 10px;">
        <img src="https://github.com/user-attachments/assets/499e8fac-b680-427b-8ac5-0ba33daec6e5" width="500" />
    </div>
    <div style="text-align: center; margin-bottom: 10px;">
        <img src="https://github.com/user-attachments/assets/37ccf09a-3007-437e-bbac-1a0f3c5a3168" width="500" />
    </div>
    <div style="text-align: center; margin-bottom: 10px;">
        <img src="https://github.com/user-attachments/assets/88aadc49-d7a9-48b3-8fb6-9ba8fb6d6384" width="500" />
    </div>
</div>
