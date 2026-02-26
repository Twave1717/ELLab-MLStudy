# ELLab ML Study
- dataset: CIFAR10
- task: image classification
- pytorch
- from scratch
    - pytorch 공식 문서 / tutorial
    - AI x
- 코드 모듈화
    - 재사용이 가능하고
    - 새로운 모듈 추가가 쉽고
    - 다양한 세팅을 코드 수정 없이 실행 가능하도록
    - 학습 loss, epoch별 test accuracy -> tensorboard로 기록하도록

# Quick Start
```bash
conda env create -f environment.yml
conda env activate MLStudy
python main.py --model resnet-20

chmod +x ./tensorboard.sh
./tensorboard.sh
```

# Models
## Resnet
- model name: resnet-n
- n: layer size (n = 6a + 2 형태의 값이어야 함)