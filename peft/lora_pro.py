"""LoRA-Pro optimizer.

LoRA A/B를 직접 업데이트하는 대신, 등가 full-matrix gradient에 Adam을 적용한 뒤
그 업데이트를 A/B 공간으로 되돌려 투영한다.

구현은 개인 실험 runner(`2SFS_All_LoRA_Methods_RunPod_v7.ipynb`의
`run_all_lora_methods_v7.py`)의 `LoRAProOptimizer`를 그대로 옮긴 것이며,
loralib의 `w_lora_A`/`w_lora_B` 대신 `peft.lora.LoRALinear`의
`lora_a.weight`/`lora_b.weight`를 사용한다는 점만 다르다.
"""

import torch


ADAM_EPS = 1e-8
MATRIX_DAMPING = 1e-8
B_TOL = 1e-12
DIAGNOSTIC_EVERY = 100


def optimizer_state_bytes(modules):
    """LoRA-Pro가 유지하는 Adam moment의 크기.

    moment는 A/B가 아니라 등가 full matrix `(out, in)` 두 개를 float32로 들고 있다.
    """
    return sum(
        2 * module.lora_b.out_features * module.lora_a.in_features * 4
        for module in modules.values()
    )


def solve_sylvester_symmetric(left, right, value, damping=MATRIX_DAMPING):
    left_values, left_vectors = torch.linalg.eigh(left.float())
    right_values, right_vectors = torch.linalg.eigh(right.float())
    rotated = left_vectors.t() @ value.float() @ right_vectors
    denominator = left_values[:, None] + right_values[None, :]
    solved = rotated / denominator.clamp_min(damping)
    return (left_vectors @ solved @ right_vectors.t()).to(value.dtype)


class LoRAProOptimizer(torch.optim.Optimizer):
    """`modules`는 `{name: LoRALinear}` dict이다.

    A/B parameter뿐 아니라 각 module의 `scaling`이 필요하므로 parameter 목록이
    아니라 module dict를 받는다. weight decay는 사용하지 않는다.
    """

    def __init__(
        self,
        modules,
        lr,
        betas=(0.9, 0.999),
        adam_eps=ADAM_EPS,
        matrix_damping=MATRIX_DAMPING,
        b_tol=B_TOL,
        diagnostic_every=DIAGNOSTIC_EVERY
    ):
        if not modules:
            raise ValueError("LoRAProOptimizer requires at least one LoRA module")

        self.pairs = []
        self.matrix_damping = float(matrix_damping)
        self.b_tol = float(b_tol)
        self.diagnostic_every = int(diagnostic_every)

        parameters = []
        for name, module in sorted(modules.items()):
            a, b = module.lora_a.weight, module.lora_b.weight
            self.pairs.append((name, a, b, module.scaling))
            parameters.extend((a, b))

        super().__init__(
            parameters,
            dict(lr=lr, betas=betas, eps=adam_eps, weight_decay=0.0)
        )
        self.param_groups[0].setdefault("optimizer_steps", 0)
        self.param_groups[0].setdefault("ascent_steps", 0)
        self.param_groups[0].setdefault("last_predicted_dloss", 0.0)
        self.param_groups[0].setdefault("max_positive_predicted_dloss", 0.0)

    @torch.no_grad()
    def step(self, closure=None):
        if closure is not None:
            with torch.enable_grad():
                closure()

        lr = self.param_groups[0]["lr"]
        beta1, beta2 = self.param_groups[0]["betas"]
        adam_eps = self.param_groups[0]["eps"]
        damping = self.matrix_damping
        predicted_dloss_total = None

        for _name, a, b, scaling in self.pairs:
            if a.grad is None or b.grad is None:
                continue

            grad_a_raw = a.grad.float()
            grad_b_raw = b.grad.float()
            a32 = a.float()
            b32 = b.float()
            aa = a32 @ a32.t()
            btb = b32.t() @ b32
            eye_r = torch.eye(a.shape[0], device=a.device, dtype=torch.float32)
            aa_inv = torch.linalg.pinv(aa + damping * eye_r)

            state = self.state[a]
            step = int(state.get("step", 0))
            diagnostic_step = step == 0 or (step + 1) % self.diagnostic_every == 0
            if diagnostic_step:
                cond_aa = torch.linalg.cond(aa + damping * eye_r).item()
                cond_btb = torch.linalg.cond(btb + damping * eye_r).item()
                state["last_cond_aa"] = cond_aa
                state["last_cond_btb"] = cond_btb
                state["max_cond_aa"] = max(float(state.get("max_cond_aa", 0.0)), cond_aa)
                state["max_cond_btb"] = max(float(state.get("max_cond_btb", 0.0)), cond_btb)

            # B는 0으로 초기화되므로 첫 step에서는 B^T B가 특이행렬이다.
            b_is_degenerate = bool(b32.norm() <= self.b_tol)
            if b_is_degenerate:
                grad_a = grad_a_raw
                grad_b = grad_b_raw @ aa_inv / (scaling ** 2)
            else:
                btb_inv = torch.linalg.pinv(btb + damping * eye_r)
                projection = torch.eye(
                    b.shape[0], device=b.device, dtype=torch.float32
                ) - b32 @ btb_inv @ b32.t()
                grad_a = btb_inv @ grad_a_raw / (scaling ** 2)
                grad_b = projection @ grad_b_raw @ aa_inv / (scaling ** 2)

            equivalent = scaling * (b32 @ grad_a + grad_b @ a32)
            if step == 0:
                state["exp_avg_full"] = torch.zeros_like(equivalent)
                state["exp_avg_sq_full"] = torch.zeros_like(equivalent)
            first = state["exp_avg_full"]
            second = state["exp_avg_sq_full"]
            first.mul_(beta1).add_(equivalent, alpha=1 - beta1)
            second.mul_(beta2).addcmul_(equivalent, equivalent, value=1 - beta2)
            step += 1
            state["step"] = step
            first_hat = first / (1 - beta1 ** step)
            second_hat = second / (1 - beta2 ** step)
            full_adam = first_hat / (second_hat.sqrt() + adam_eps)

            projected_a = scaling * b32.t() @ full_adam
            projected_b = scaling * full_adam @ a32.t()
            if b_is_degenerate:
                final_a = projected_a
                final_b = projected_b @ aa_inv / (scaling ** 2)
            else:
                btb_inv = torch.linalg.pinv(btb + damping * eye_r)
                x_value = -(btb_inv @ projected_a @ a32.t()) / (scaling ** 2)
                x = solve_sylvester_symmetric(btb, aa, x_value, damping)
                if diagnostic_step:
                    x_norm = x.norm().item()
                    state["last_x_norm"] = x_norm
                    state["max_x_norm"] = max(float(state.get("max_x_norm", 0.0)), x_norm)
                projection = torch.eye(
                    b.shape[0], device=b.device, dtype=torch.float32
                ) - b32 @ btb_inv @ b32.t()
                final_a = btb_inv @ projected_a / (scaling ** 2) + x @ a32
                final_b = projection @ projected_b @ aa_inv / (scaling ** 2) - b32 @ x

            predicted_dloss = -lr * (
                (grad_a_raw * final_a).sum() + (grad_b_raw * final_b).sum()
            )
            predicted_dloss_total = (
                predicted_dloss
                if predicted_dloss_total is None
                else predicted_dloss_total + predicted_dloss
            )
            a.add_(final_a.to(a), alpha=-lr)
            b.add_(final_b.to(b), alpha=-lr)

        if predicted_dloss_total is not None:
            predicted_value = float(predicted_dloss_total.item())
            group = self.param_groups[0]
            group["optimizer_steps"] = int(group.get("optimizer_steps", 0)) + 1
            group["last_predicted_dloss"] = predicted_value
            if predicted_value > 0:
                group["ascent_steps"] = int(group.get("ascent_steps", 0)) + 1
                group["max_positive_predicted_dloss"] = max(
                    float(group.get("max_positive_predicted_dloss", 0.0)),
                    predicted_value
                )
        return None

    def diagnostics(self):
        values = [self.state[a] for _name, a, _b, _scaling in self.pairs if a in self.state]
        group = self.param_groups[0]
        optimizer_steps = int(group.get("optimizer_steps", 0))
        ascent_steps = int(group.get("ascent_steps", 0))
        return {
            "max_cond_aa": max(
                (float(value.get("max_cond_aa", 0.0)) for value in values), default=0.0
            ),
            "max_cond_btb": max(
                (float(value.get("max_cond_btb", 0.0)) for value in values), default=0.0
            ),
            "max_x_norm": max(
                (float(value.get("max_x_norm", 0.0)) for value in values), default=0.0
            ),
            "optimizer_steps": optimizer_steps,
            "ascent_steps": ascent_steps,
            "ascent_fraction": ascent_steps / optimizer_steps if optimizer_steps else 0.0,
            "last_predicted_dloss": float(group.get("last_predicted_dloss", 0.0)),
            "max_positive_predicted_dloss": float(
                group.get("max_positive_predicted_dloss", 0.0)
            ),
            "matrix_damping": self.matrix_damping,
            "adam_eps": float(group["eps"]),
        }
