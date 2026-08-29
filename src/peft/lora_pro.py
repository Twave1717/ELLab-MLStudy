"""LoRA-Pro optimizer."""

import torch


ADAM_EPS = 1e-8
MATRIX_DAMPING = 1e-8
B_TOL = 1e-12


def solve_sylvester_symmetric(left, right, value, damping=MATRIX_DAMPING):
    left_values, left_vectors = torch.linalg.eigh(left.float())
    right_values, right_vectors = torch.linalg.eigh(right.float())
    rotated = left_vectors.t() @ value.float() @ right_vectors
    denominator = left_values[:, None] + right_values[None, :]
    solved = rotated / denominator.clamp_min(damping)
    return (left_vectors @ solved @ right_vectors.t()).to(value.dtype)


class LoRAProOptimizer(torch.optim.Optimizer):
    """Optimize LoRA pairs through full-matrix moments."""

    def __init__(
        self,
        modules,
        lr,
        betas=(0.9, 0.999),
        adam_eps=ADAM_EPS,
        matrix_damping=MATRIX_DAMPING,
        b_tol=B_TOL,
    ):
        self.pairs = []
        self.matrix_damping = matrix_damping
        self.b_tol = b_tol

        parameters = []
        for module in modules.values():
            a, b = module.lora_a.weight, module.lora_b.weight
            self.pairs.append((a, b, module.scaling))
            parameters.extend((a, b))

        super().__init__(
            parameters,
            dict(lr=lr, betas=betas, eps=adam_eps, weight_decay=0.0),
        )

    @torch.no_grad()
    def step(self, closure=None):
        if closure is not None:
            with torch.enable_grad():
                closure()

        lr = self.param_groups[0]["lr"]
        beta1, beta2 = self.param_groups[0]["betas"]
        adam_eps = self.param_groups[0]["eps"]

        for a, b, scaling in self.pairs:
            if a.grad is None or b.grad is None:
                continue

            grad_a_raw = a.grad.float()
            grad_b_raw = b.grad.float()
            a32 = a.float()
            b32 = b.float()
            aa = a32 @ a32.t()
            btb = b32.t() @ b32
            eye = torch.eye(a.shape[0], device=a.device, dtype=torch.float32)
            aa_inv = torch.linalg.pinv(aa + self.matrix_damping * eye)
            b_is_degenerate = bool(b32.norm() <= self.b_tol)

            if b_is_degenerate:
                grad_a = grad_a_raw
                grad_b = grad_b_raw @ aa_inv / scaling ** 2
            else:
                btb_inv = torch.linalg.pinv(btb + self.matrix_damping * eye)
                projection = (
                    torch.eye(b.shape[0], device=b.device, dtype=torch.float32)
                    - b32 @ btb_inv @ b32.t()
                )
                grad_a = btb_inv @ grad_a_raw / scaling ** 2
                grad_b = projection @ grad_b_raw @ aa_inv / scaling ** 2

            equivalent = scaling * (b32 @ grad_a + grad_b @ a32)
            state = self.state[a]
            step = state.get("step", 0) + 1
            if step == 1:
                state["exp_avg_full"] = torch.zeros_like(equivalent)
                state["exp_avg_sq_full"] = torch.zeros_like(equivalent)
            first = state["exp_avg_full"]
            second = state["exp_avg_sq_full"]
            first.mul_(beta1).add_(equivalent, alpha=1 - beta1)
            second.mul_(beta2).addcmul_(equivalent, equivalent, value=1 - beta2)
            state["step"] = step
            update = (
                first / (1 - beta1 ** step)
                / ((second / (1 - beta2 ** step)).sqrt() + adam_eps)
            )

            projected_a = scaling * b32.t() @ update
            projected_b = scaling * update @ a32.t()
            if b_is_degenerate:
                final_a = projected_a
                final_b = projected_b @ aa_inv / scaling ** 2
            else:
                btb_inv = torch.linalg.pinv(btb + self.matrix_damping * eye)
                x_value = -(btb_inv @ projected_a @ a32.t()) / scaling ** 2
                x = solve_sylvester_symmetric(
                    btb, aa, x_value, self.matrix_damping
                )
                projection = (
                    torch.eye(b.shape[0], device=b.device, dtype=torch.float32)
                    - b32 @ btb_inv @ b32.t()
                )
                final_a = btb_inv @ projected_a / scaling ** 2 + x @ a32
                final_b = (
                    projection @ projected_b @ aa_inv / scaling ** 2 - b32 @ x
                )

            a.add_(final_a.to(a), alpha=-lr)
            b.add_(final_b.to(b), alpha=-lr)
