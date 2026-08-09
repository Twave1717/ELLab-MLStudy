import torch
import torch.nn.functional as F
from torch import nn


class TwoStageCLIP(nn.Module):
    def __init__(self, clip_model, tokenizer, classnames, template):
        super().__init__()
        classnames = [classname[0].replace("_", " ") for classname in classnames]
        tokens = tokenizer(
            [template.format(name) for name in classnames],
            padding=True,
            return_tensors="pt"
        )

        self.model = clip_model
        self.tokenizer = tokenizer
        self.template = template
        self.register_buffer("input_ids", tokens["input_ids"], persistent=False)
        self.register_buffer("attention_mask", tokens["attention_mask"], persistent=False)
        self.classifier = None

    def encode_image(self, images):
        vision_outputs = self.model.vision_model(pixel_values=images)
        features = self.model.visual_projection(vision_outputs.pooler_output)
        return F.normalize(features, dim=-1)

    def encode_text(self):
        outputs = self.model.text_model(
            input_ids=self.input_ids,
            attention_mask=self.attention_mask
        )
        features = self.model.text_projection(outputs.pooler_output)
        return F.normalize(features, dim=-1)

    def encode_classnames(self, classnames):
        names = [classname[0].replace("_", " ") for classname in classnames]
        tokens = self.tokenizer([self.template.format(name) for name in names], padding=True, return_tensors="pt").to(self.input_ids.device)
        outputs = self.model.text_model(input_ids=tokens["input_ids"], attention_mask=tokens["attention_mask"])
        features = self.model.text_projection(outputs.pooler_output)
        return F.normalize(features, dim=-1)

    def stage_one_logits(self, images):
        image_features = self.encode_image(images)
        text_features = self.encode_text()
        return self.model.logit_scale.exp() * image_features @ text_features.t()

    def initialize_classifier(self):
        self.model.eval()
        with torch.no_grad():
            text_features = self.encode_text()
        self.model.requires_grad_(False)
        self.classifier = nn.Parameter(text_features)

    def classifier_logits(self, images, classifier):
        with torch.no_grad():
            image_features = self.encode_image(images)
        classifier = F.normalize(classifier, dim=-1)
        return self.model.logit_scale.exp() * image_features @ classifier.t()

    def stage_two_logits(self, images):
        return self.classifier_logits(images, self.classifier)
