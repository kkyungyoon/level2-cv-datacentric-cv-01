# BoostCamp AI Tech 7th CV-01 Object Detection Project


## Team Members
<div align="center">
<table>
<tr>
<td><img src="https://github.com/user-attachments/assets/539241ed-cfcd-4055-8c9f-0766df482995" width="100"/></td>
<td><img src="https://github.com/user-attachments/assets/aa787599-06e3-4eda-ab0d-b234c8633a26" width="100"/></td>
<td><img src="https://github.com/user-attachments/assets/776e4f23-c1dc-46b0-b56b-8b5418e46c53" width="100"/></td>
<td><img src="https://github.com/user-attachments/assets/2814cffd-5219-4dab-929f-ebd48d8fa4d3" width="100"/></td>
<td><img src="https://github.com/user-attachments/assets/e84a8972-9e37-426d-85c7-501d68a1a24e" width="100"/></td>
<td><img src="https://github.com/user-attachments/assets/f9589259-b6bd-46d0-96b6-e2e8a91741d8" width="100"/></td>
</tr>
<tr>
<td><b>동준</b></td>
<td><b>경윤</b></td>
<td><b>영석</b></td>
<td><b>태영</b></td>
<td><b>태성</b></td>
<td><b>세린</b></td>
</tr>
</table>
</div>
              

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

### Final Results (Public, Private)
<div align="center">
    <img src="https://github.com/user-attachments/assets/3cf8c761-0227-4e77-9d6a-b7da305f6ff6" />
    <p style="font-weight: bold; margin-top: 10px;">
        The result of Ensemble (WBF, IoU=0.3):<br>
        Super resolution (x4) + Normalize (base) + Remove dash + 3 folds (9:1 train-valid split)<br>
        Super resolution (x4) + Normalize (custom) + Remove dash
    </p>
</div>

### Data augmentation

<div>
    <div style="text-align: center; margin-bottom: 10px;">
        <img src="https://github.com/user-attachments/assets/06200793-a5f1-4f61-9346-63a4d26dc070" width="500" />
    </div>
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
