# 2022.ComputerVision.TermProject.C3D

본 문서에서는 2022년 봄학기 컴퓨터 비전 수업의 텀 프로젝트에 대해 다룹니다.

논문 : [Learning Spatiotemporal Features with 3D Convolutional Networks](https://arxiv.org/pdf/1412.0767v4.pdf)

논문 설명 영상 : [YouTube](https://youtu.be/U6PmduQ1BQw)

eval_ai 주소 : [Link](http://203.250.148.128:3088/web/challenges/challenge-page/85/overview)

코드 설명 영상 : [YouTube](https://youtu.be/0senzUXK_3E)

# VideoClassification with C3D
Learning Spatiotemporal Features with 3D Convolutional Networks(이하 C3D)를 이용하여 수행할 video classification은 비디오를 보고, 비디오의 라벨을 예측하는 task 입니다. 본 논문에서는 video classification을 Sports-1M 데이터셋으로 학습하고, action recognition을 UCF-101로 평가합니다. Sports-1M의 학습 시간과 데이터셋 용량 문제로, 이번 텀프로젝트에서는 UCF-101만을 이용해서 학습 및 평가를 수행합니다.

## Dataset
본 프로젝트에서는 [UCF101](https://www.crcv.ucf.edu/data/UCF101.php)를 이용하여 학습 및 평가를 수행합니다.

![image](https://user-images.githubusercontent.com/24697575/167851410-3e802e6d-02f1-4fc8-aa28-86799016181f.png)

UCF101의 공식 홈페이지에서 위와 같은 부분을 찾아, 데이터셋을 다운로드 받아주세요. 다운로드된 데이터셋의 경로를 mypath.py에 입력합니다.

Train/Test split의 경우에는 Action Recognition을 다운로드 받아 사용해주세요.

주의할 점은 "Train/Test split"이 3가지가 존재합니다.

- testlist01.txt & trainlist01.txt (베이스라인)
- testlist02.txt & trainlist02.txt
- testlist03.txt & trainlist03.txt

01, 02, 03 중에서 여기서는 01 만을 사용합니다. 학습에서 3가지 split을 함께 사용하지 않도로 주의해주세요. "annotation" 폴더에 넣어둔 파일을 사용해도 됩니다. (3가지 모두 사용할 경우, train/test 비디오가 서로 겹치게됩니다.)

## Requirements

본 프로젝트에서는 [pytorch-video-recognition](https://github.com/jfzhang95/pytorch-video-recognition)을 수정하여 사용합니다.

python3.5 이상의 환경에서 각자 환경에서 GPU를 지원하는 버전의 pytorch를 설치한 후에 아래 패키지를 추가로 설치해주세요.

```
conda install opencv
pip install tqdm scikit-learn numpy pandas natsort 
```

## 데이터 구조

먼저 데이터가 올바르게 구성되었는지 설정이 필요합니다. my_path.py에서 각 경로를 올바르게 설정했는지 확인해주세요.

```
UCF-101
├── ApplyEyeMakeup
│   ├── v_ApplyEyeMakeup_g01_c01.avi
│   └── ...
├── ApplyLipstick
│   ├── v_ApplyLipstick_g01_c01.avi
│   └── ...
└── Archery
│   ├── v_Archery_g01_c01.avi
│   └── ...
```
UCF101의 형태는 위와 같으면 됩니다.

```
ucf101
├── ApplyEyeMakeup
│   ├── v_ApplyEyeMakeup_g01_c01
│   │   ├── 00001.jpg
│   │   └── ...
│   └── ...
├── ApplyLipstick
│   ├── v_ApplyLipstick_g01_c01
│   │   ├── 00001.jpg
│   │   └── ...
│   └── ...
└── Archery
│   ├── v_Archery_g01_c01
│   │   ├── 00001.jpg
│   │   └── ...
│   └── ...
```
빠른 학습을 위해 데이터 전처리 과정이 포함되어 있습니다. 전처리 과정이 끝난 뒤에는 위와 같은 구조를 가지게 됩니다.

데이터 로드 및 전처리 과정에 문제가 있다면 위 부분을 참고하세요.

## How to run
모델을 학습하기 위해서는 아래 코드를 실행해주세요.
```
python train.py --useTest
```

만약 최초로 실행하는 경우라면, 아래의 코드를 실행해주세요.
```
python train.py --preprocess --useTest
```


평가는 아래 코드를 실행해주세요.
```
python evaluation.py --trained_model_path /path/to/trained_model
```

## Metric

평가는 [ACC](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html?highlight=acc#sklearn.metrics.accuracy_score)를 이용합니다.

## Performance

||train_ACC|test_ACC|
|:---:|:---:|:---:|
|C3D_UCF|98.469%|42.162	%|
