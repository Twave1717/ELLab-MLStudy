# ELLab ML Study

PyTorch로 image classification과 self-supervised learning 방법을 구현하고
비교하기 위한 프로젝트입니다.

## Quick Start

```bash
# install dependencies
uv sync
# run
uv run python train.py # 기본 설정은 CIFAR-10, supervised learning, ResNet-20
```

```bash
# 모델과 데이터셋을 직접 선택
uv run python train.py \
  --method supervised \
  --model resnet-20 \
  --dataset cifar10
```

```bash
# CLIP을 학습하려면 `--method clip`만 지정
# `openai/clip-vit-base-patch16`이 자동으로 선택 (pre-trained)
uv run python train.py \
  --method clip \
  --dataset cifar10 \
  --peft lora
```

```bash
# 2SFS LayerNorm
uv run python train_2sfs.py \
  --dataset cifar10 \
  --shots 1

# LoRA를 사용하려면 추가
uv run python train_2sfs.py \
  --dataset cifar10 \
  --shots 1 \
  --peft lora
```

```bash
# `train.py` 학습 기록은 `runs/`에 저장
chmod +x tensorboard.sh
./tensorboard.sh
```

Kaggle 3종 데이터셋의 optimized breakpoint는 다음처럼 탐색합니다.

```bash
uv run python notebook/find_optimized_breakpoint.py \
  --dataset eurosat \
  --shots 16 \
  --peft ln \
  --gradient_gate abs_identity
```

스크립트는 exact Kaggle Base/Novel split의 전체 validation Novel accuracy가 최초로
최대가 되는 step을 기본 breakpoint로 저장합니다. Base, Novel, HM의 모든 probe 기록은
JSON에 남기고 같은 위치에 논문형 trajectory PNG도 생성합니다. 현재 Dynamic Gate의
AMP는 기본적으로 꺼져 있습니다. 현재 Dynamic Gate의 AMP overflow 처리는 보완
전이므로 재현성을 우선하는 실험에서는 `--amp`를 추가하지 않습니다.
`notebook/01_find_breakpoints.ipynb`와
`notebook/02_compare_breakpoint_ratios.ipynb`는 기존 실험 흐름과 fixed-ratio 비교를
설명하는 참고 노트북입니다. Manifest 로딩은 `notebook/utils/data.py`, 공통 학습·평가는
`notebook/utils/experiment.py`를 재사용합니다.

저장된 JSON에서 figure를 다시 만들거나 수동 선택선을 추가할 수도 있습니다.

```bash
uv run python notebook/utils/breakpoint_plot.py \
  archive/05_optimized_breakpoint_experiments/optimized_breakpoint_eurosat_ln_abs_identity_16shot.json \
  --manual-step 1200
```

## Methods

| Method | 지원 architecture | 비고 |
| --- | --- | --- |
| `supervised` | ResNet, PreActResNet, DenseNet, MLP-Mixer, ViT, RotNet | FractalNet 제외 |
| `byol` | ResNet, PreActResNet, DenseNet, MLP-Mixer, ViT, RotNet | FractalNet 제외 |
| `simclr` | ResNet, PreActResNet, DenseNet, MLP-Mixer, ViT, RotNet | FractalNet 제외 |
| `moco` | ResNet, PreActResNet, DenseNet, MLP-Mixer, ViT, RotNet | FractalNet 제외 |
| `rotnet` | ResNet, PreActResNet, DenseNet, MLP-Mixer, ViT, RotNet | FractalNet 제외 |
| `clip` | CLIP ViT-B/16 | Hugging Face pretrained model 고정 |
| `2sfs` | CLIP ViT-B/16 | `train_2sfs.py`에서 실행 |

## Evaluation

```bash
# CLIP zero-shot evaluation
uv run python evaluate.py clip cifar10 \
  --batch_size 64 \
  --data_root data
```

```bash
# RotNet pretraining
uv run python train.py \
  --method rotnet \
  --model rotnet-4 \
  --dataset cifar10
```

```bash
# RotNet nonlinear evaluation
uv run python evaluate.py rotnet \
  checkpoint/.../encoder.pth
```

CLIP zero-shot은 지정한 데이터셋의 test split을 사용합니다. RotNet
evaluation은 현재 `rotnet-4`와 CIFAR-10만 지원합니다.

## Architectures

모델 architecture, 학습 method, evaluation head, PEFT 구현은 각각
`src/architecture`, `src/methods`, `src/eval`, `src/peft`에 있습니다.

### From scratch

다음은 프로젝트에 등록된 from-scratch architecture입니다.

```text
resnet-20
preactresnet-20
densenet-40
fractalnet-20
fractalnet-40
fractalnet_droppath-20
fractalnet_droppath-40
mlp_mixer-12
vit-12
rotnet-4
```

| From-scratch 미지원 구현 | 비고 |
| --- | --- |
| `fractalnet-20` | 현재 설계 수정 필요 |
| `fractalnet-40` | 현재 설계 수정 필요 |
| `fractalnet_droppath-20` | 현재 설계 수정 필요 |
| `fractalnet_droppath-40` | 현재 설계 수정 필요 |
| 직접 구현한 PyTorch CLIP | 미구현, Hugging Face CLIP 사용 |
| 직접 구현한 PyTorch ViT | 미구현, Hugging Face ViT 사용 |

`resnet-20`은 작은 이미지를 위한 CIFAR-style ResNet입니다.

### Torchvision ResNet

```text
resnet-18
resnet-34
resnet-50
resnet-101
resnet-152
```

```bash
uv run python train.py \
  --model resnet-18 \
  --pretrained
```

### ViT

`vit-12`는 Hugging Face `ViTModel`을 사용합니다. `--pretrained`를 지정하면
`google/vit-base-patch16-224-in21k` weight를 가져옵니다.

```bash
uv run python train.py \
  --model vit-12 \
  --pretrained
```

`--pretrained`는 torchvision ResNet과 `vit-12`만 지원합니다.

## CLIP

CLIP은 Hugging Face의 pretrained `openai/clip-vit-base-patch16`을
사용합니다. 별도의 `--pretrained` 옵션은 사용하지 않습니다.

현재 학습 범위는 다음과 같습니다.

| 구성 요소 | 일반 CLIP | CLIP + LoRA |
| --- | --- | --- |
| Vision encoder | 학습 | 기존 weight 고정 |
| Vision QKV LoRA | 없음 | 학습 |
| Visual projection | 학습 | 고정 |
| Text encoder | 고정 | 고정 |
| Text projection | 고정 | 고정 |
| Logit scale | 학습 | 고정 |

CLIP은 각 데이터셋 파일의 `template`과 class name으로 prompt를 만듭니다.
예를 들어 UCF101은 `a photo of a person doing {}.`을 사용합니다.
Text encoder가 고정되어 있으므로 class prompt embedding은 모델 생성 시 한
번만 계산해 buffer에 보관하고 모든 학습 및 평가 batch에서 재사용합니다.

### LoRA

LoRA는 vision encoder 모든 Transformer layer의 `q_proj`, `k_proj`,
`v_proj`에만 적용합니다.

```text
rank = 2
alpha = 1
dropout = 0.25
```

이 값은 `rethinking_fewshot_vlms`의 기본 설정을 사용합니다. LoRA 실행 시
기존 CLIP parameter는 모두 고정되고 LoRA parameter만 학습합니다.

## Datasets

기본 데이터 경로는 `data/`이며 `--data_root`로 변경할 수 있습니다.
각 데이터셋은 `data/<dataset-name>/` 아래에 저장됩니다. 예를 들어
CIFAR-10은 `data/cifar10/cifar-10-batches-py`, Imagenette는
`data/imagenette/imagenette2-160`을 사용합니다.

| 이름 | torchvision dataset | 다운로드 |
| --- | --- | --- |
| `cifar10` | CIFAR10 | 자동 |
| `caltech101` | Caltech101 | 자동 |
| `dtd` | DTD | 자동 |
| `eurosat` | EuroSAT | 자동 |
| `fgvc` | FGVCAircraft | 자동 |
| `food101` | Food101 | 자동 |
| `imagenette` | Imagenette | 자동 |
| `oxford_flowers` | Flowers102 | 자동 |
| `oxford_pets` | OxfordIIITPet | 자동 |
| `stanford_cars` | StanfordCars | 자동 |
| `sun397` | SUN397 | 자동 |
| `imagenet`, `imagenet-ilsvrc2012` | ImageNet | 수동 |
| `ucf101` | UCF101 | 수동 |

ImageNet은 `data/imagenet`, UCF101은 `data/ucf101` 아래에 원본 파일을
직접 준비해야 합니다. UCF101은 video classification 대신 각 clip의 첫
frame을 image classification 입력으로 사용합니다.

공식 validation split이 있는 DTD, FGVCAircraft, Flowers102는 이를 그대로
사용합니다. 공식 test split만 있는 데이터셋은 train의 10%를 validation으로
분리합니다. 별도 split이 없는 Caltech101, EuroSAT, SUN397은 70/10/20으로
나눕니다. 모든 무작위 분할과 학습은 `GLOBAL_SEED = 2026`을 사용합니다.

## Preprocessing

| 설정 | Crop size | Normalize | Interpolation |
| --- | --- | --- | --- |
| 기본 | 데이터셋 설정값 | mean/std 0.5 | Bilinear |
| Pretrained ResNet | 224 | ImageNet mean/std | Bilinear |
| CLIP | 224 | CLIP mean/std | Bicubic |

CIFAR-10의 기본 crop size는 32, Imagenette는 160, 나머지는 224입니다.
BYOL, SimCLR, MoCo는 같은 transform을 두 번 적용해 두 개의 view를
생성합니다.

## Training Options

| Option | 기본값 | 설명 |
| --- | --- | --- |
| `--method` | `supervised` | 학습 방법 |
| `--model` | method에 따라 선택 | 일반 method는 `resnet-20`, CLIP은 pretrained CLIP |
| `--dataset` | `cifar10` | 데이터셋 |
| `--peft` | `none` | `none` 또는 `lora` |
| `--batch_size` | `128` | Batch size |
| `--epochs` | `200` | Epoch 수 |
| `--lr` | method에 따라 선택 | CLIP `2e-4`, 나머지 `0.1` |
| `--weight_decay` | method에 따라 선택 | CLIP `1e-2`, 나머지 `1e-4` |
| `--scheduler` | method에 따라 선택 | CLIP cosine, 나머지 multistep |
| `--grad_clip` | 없음 | Gradient norm 제한 |
| `--pretrained` | `false` | ResNet 또는 ViT pretrained weight 사용 |
| `--data_root` | `data` | 데이터 저장 경로 |
| `--save_path` | `checkpoint` | 학습 결과 저장 경로 |

일반 method는 SGD와 momentum 0.9를 사용합니다. CLIP은 AdamW를
사용합니다. 매 epoch의 validation accuracy가 가장 높은 checkpoint를
`best_` prefix로 저장합니다. 학습 종료 후 마지막 checkpoint도 별도로 저장합니다.

```bash
# 전체 옵션 예시
uv run python train.py \
  --method clip \ # supervised, byol, simclr, rotnet, moco, clip
  --model openai/clip-vit-base-patch16 \ # resnet-20, preactresnet-20, densenet-40, fractalnet-20/40, fractalnet_droppath-20/40, mlp_mixer-12, vit-12, rotnet-4, resnet-18/34/50/101/152
  --dataset cifar10 \ # caltech101, dtd, eurosat, fgvc, food101, imagenet, imagenet-ilsvrc2012, imagenette, oxford_flowers, oxford_pets, stanford_cars, sun397, ucf101
  --peft lora \ # none, lora
  --pretrained \ # torchvision ResNet 또는 vit-12에서 사용, CLIP에서는 사용하지 않음
  --epochs 200 \ # epoch 수
  --batch_size 128 \ # batch size
  --lr 0.0002 \ # CLIP 기본값 2e-4, 나머지 0.1
  --weight_decay 0.01 \ # CLIP 기본값 1e-2, 나머지 1e-4
  --scheduler cosine \ # multistep, cosine
  --grad_clip 1.0 \ # gradient norm 제한
  --data_root data \ # dataset 경로
  --save_path checkpoint # 학습 결과 경로
```

## Outputs

일반 학습 결과:

```text
checkpoint/<model>-<method>-<epochs>-<time>/
├── encoder.pth
├── method.pth
├── best_encoder.pth
└── best_method.pth
```

LoRA 학습 결과:

```text
checkpoint/<model>-clip-<epochs>-<time>/
├── lora.pth
└── best_lora.pth
```

LoRA checkpoint에는 adapter weight와 사용한 CLIP model 이름만
저장합니다. 현재 CLI에는 checkpoint를 다시 불러오는 기능이 없습니다.

## Current Scope

- Dataset, CLIP, vision LoRA 구현
- RotNet nonlinear evaluation 구현
- CLIP architecture는 Hugging Face 구현 사용
- PyTorch 기반 CLIP, ViT architecture 직접 구현은 미완료
- all-to-all 2SFS-LayerNorm/LoRA 구현
- 일반 CLIP Text LoRA와 QLoRA 미구현
