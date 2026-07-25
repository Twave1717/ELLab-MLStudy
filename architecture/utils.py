from torch import nn
from torchvision.models import get_model

from .densenet import DenseNet
from .fractalnet import FractalNet, FractalNetDropPath
from .mlp_mixer import MLPMixer
from .preactresnet import PreActResNet
from .resnet import ResNet
from .rotnet import RotNet
from .vit import VisionTransformer


_ENCODERS = {
    "resnet": ResNet,
    "preactresnet": PreActResNet,
    "densenet": DenseNet,
    "fractalnet": FractalNet,
    "fractalnet_droppath": FractalNetDropPath,
    "mlp_mixer": MLPMixer,
    "vit": VisionTransformer,
    "rotnet": RotNet,
}

_TORCHVISION_RESNETS = (
    "resnet-18",
    "resnet-34",
    "resnet-50",
    "resnet-101",
    "resnet-152",
)

CLIP_MODEL = "openai/clip-vit-base-patch16"

MODEL_CHOICES = (
    "resnet-20",
    "preactresnet-20",
    "densenet-40",
    "fractalnet-20",
    "fractalnet-40",
    "fractalnet_droppath-20",
    "fractalnet_droppath-40",
    "mlp_mixer-12",
    "vit-12",
    "rotnet-4",
    *_TORCHVISION_RESNETS,
    CLIP_MODEL,
)


def build_encoder(model_name, pretrained=False):
    if model_name in _TORCHVISION_RESNETS:
        return _build_resnet(model_name, pretrained)

    name, num_layers = model_name.rsplit("-", 1)
    try:
        encoder = _ENCODERS[name]
    except KeyError:
        raise ValueError(f"Unknown model: {model_name}") from None
    if name == "vit":
        return encoder(int(num_layers), pretrained=pretrained)
    if pretrained:
        raise ValueError(
            f"{model_name} has no built-in pretrained weights"
        )
    return encoder(int(num_layers))


def _build_resnet(model_name, pretrained):
    model_name = model_name.replace("-", "")

    if pretrained:
        model = get_model(model_name, weights="DEFAULT")
    else:
        model = get_model(model_name)
    classifier = model.fc
    model.num_features = classifier.in_features
    model.fc = nn.Identity()
    return model


def load_clip(model_name):
    from transformers import AutoTokenizer, CLIPModel

    model = CLIPModel.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer
