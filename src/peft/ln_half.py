from torch import nn


def mark_only_half_layernorm_as_trainable(clip_model):
    clip_model.requires_grad_(False)

    for encoder in (clip_model.vision_model, clip_model.text_model):
        idx = 0
        for module in encoder.modules():
            if isinstance(module, nn.LayerNorm):
                if idx % 2 == 1:
                    module.requires_grad_(True)

                idx += 1

    return [parameter for parameter in clip_model.parameters() if parameter.requires_grad]
