import torch
import torch.nn as nn
import torch.nn.functional as F
from methods.twostage import TwoStageCLIP

class PromptLearner(nn.Module):
    def __init__(self, clip_model, tokenizer, classnames, template, n_ctx=8):
        super().__init__()
        self.n_ctx = n_ctx
        self.tokenizer = tokenizer
        
        # Hugging Face CLIP 토큰 임베딩
        self.token_embedding = clip_model.text_model.embeddings.token_embedding
        ctx_dim = self.token_embedding.embedding_dim
        
        # Context Vector 초기화 (N(0, 0.02))
        ctx_vectors = torch.empty(n_ctx, ctx_dim)
        nn.init.normal_(ctx_vectors, std=0.02)
        self.ctx = nn.Parameter(ctx_vectors)
        
        # 클래스명 추출 및 템플릿 준비
        names = [name.replace("_", " ") if isinstance(name, str) else name[0].replace("_", " ") for name in classnames]
        dummy_prefix = " ".join(["X"] * n_ctx)
        prompts = [f"{dummy_prefix} {name}." for name in names]
        
        # 프롬프트 토큰화
        tokens = tokenizer(prompts, padding=True, return_tensors="pt")
        self.register_buffer("input_ids", tokens["input_ids"], persistent=False)
        self.register_buffer("attention_mask", tokens["attention_mask"], persistent=False)
        
        # KgCoOp의 Discrepancy Loss를 위한 Zero-shot Text Features 사전 계산
        with torch.no_grad():
            zs_prompts = [template.format(name) for name in names]
            zs_tokens = tokenizer(zs_prompts, padding=True, return_tensors="pt")
            zs_outputs = clip_model.text_model(
                input_ids=zs_tokens["input_ids"].to(clip_model.device), 
                attention_mask=zs_tokens["attention_mask"].to(clip_model.device)
            )
            zs_features = clip_model.text_projection(zs_outputs.pooler_output)
            self.register_buffer("zs_text_features", F.normalize(zs_features, dim=-1), persistent=False)

    def get_inputs_embeds(self):
        embeddings = self.token_embedding(self.input_ids)
        prefix = embeddings[:, :1, :]  # [BOS] 토큰
        suffix = embeddings[:, 1 + self.n_ctx:, :]  # 클래스명 및 [EOS] 이후 토큰들
        ctx = self.ctx.unsqueeze(0).expand(embeddings.shape[0], -1, -1)
        
        inputs_embeds = torch.cat([prefix, ctx, suffix], dim=1)
        return inputs_embeds, self.attention_mask

class _CustomTokenEmbedding(nn.Module):
    def __init__(self, custom_embeds):
        super().__init__()
        self.custom_embeds = custom_embeds
    def forward(self, input_ids):
        return self.custom_embeds

class TwoStageKgCoOp(TwoStageCLIP):
    def __init__(self, clip_model, tokenizer, classnames, template, n_ctx=8, w=8.0):
        super().__init__(clip_model, tokenizer, classnames, template)
        self.w = w
        self.prompt_learner = PromptLearner(clip_model, tokenizer, classnames, template, n_ctx)

    def encode_learned_text(self):
        inputs_embeds, attention_mask = self.prompt_learner.get_inputs_embeds()
        input_ids = self.prompt_learner.input_ids
        
        orig_token_embedding = self.model.text_model.embeddings.token_embedding
        
        try:
            self.model.text_model.embeddings.token_embedding = _CustomTokenEmbedding(inputs_embeds)
            outputs = self.model.text_model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
        finally:
            self.model.text_model.embeddings.token_embedding = orig_token_embedding
            
        features = self.model.text_projection(outputs.pooler_output)
        return F.normalize(features, dim=-1)

    # 통합을 위해 상속 구조 내에서 범용적으로 동작하도록 수정
    def stage_one_logits(self, images):
        image_features = self.encode_image(images)
        text_features = self.encode_learned_text()
        return self.model.logit_scale.exp() * image_features @ text_features.t()

    def compute_loss(self, images, labels, stage):
        if stage == "stage1":
            image_features = self.encode_image(images)
            text_features = self.encode_learned_text()
            
            logits = self.model.logit_scale.exp() * image_features @ text_features.t()
            ce_loss = F.cross_entropy(logits, labels)
            
            zs_text_features = self.prompt_learner.zs_text_features
            cos = torch.nn.CosineSimilarity(dim=1, eps=1e-07)
            score = cos(text_features, zs_text_features)
            kg_loss = 1.0 - torch.mean(score)
            
            loss = ce_loss + self.w * kg_loss
            return loss, logits
        else:
            logits = self.stage_two_logits(images)
            loss = F.cross_entropy(logits, labels)
            return loss, logits

    def initialize_classifier(self):
        self.model.eval()
        with torch.no_grad():
            text_features = self.encode_learned_text()
            
        self.model.requires_grad_(False)
        self.prompt_learner.requires_grad_(False)
        self.classifier = nn.Parameter(text_features)
