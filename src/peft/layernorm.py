from torch import nn


def mark_only_layernorm_as_trainable(clip_model):
    clip_model.requires_grad_(False)

    for encoder in (clip_model.vision_model, clip_model.text_model):
        for module in encoder.modules():
            if isinstance(module, nn.LayerNorm):
                module.requires_grad_(True)

    return [parameter for parameter in clip_model.parameters() if parameter.requires_grad]
