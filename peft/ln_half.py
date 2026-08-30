from torch import nn
import re

def mark_only_half_layernorm_as_trainable(clip_model):
    clip_model.requires_grad_(False)


    for encoder in (clip_model.vision_model, clip_model.text_model):
        for name, module in encoder.named_modules():
            if isinstance(module, nn.LayerNorm):                    
                m = re.search(r"layers\.(\d+)\.", name)

                if m and int(m.group(1)) % 2 == 1:
                    module.requires_grad_(True)
                    print(name)

    return [parameter for parameter in clip_model.parameters() if parameter.requires_grad]