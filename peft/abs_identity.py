import random

import torch
import torch.nn.functional as F


class AbsIdentityGate:
    """Scale each LayerNorm update by q = E[|g|]^2 / E[g^2]."""

    def __init__(
        self,
        parameters,
        init_images=200,
        online_images=4,
        beta=0.95,
        epsilon=1e-30,
    ):
        self.parameters = list(parameters)
        self.init_images = init_images
        self.online_images = online_images
        self.beta = beta
        self.epsilon = epsilon
        self.first_moment = None
        self.second_square = None

    def _flat_gradient(self, loss, retain_graph):
        gradients = torch.autograd.grad(
            loss,
            self.parameters,
            retain_graph=retain_graph,
            allow_unused=True,
        )
        return torch.cat([
            torch.zeros_like(parameter).flatten()
            if gradient is None else gradient.flatten()
            for gradient, parameter in zip(gradients, self.parameters)
        ])

    def initialize(self, logits_fn, dataset, device):
        python_state = random.getstate()
        torch_state = torch.random.get_rng_state()
        cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        generator = torch.Generator().manual_seed(9137)
        count = min(self.init_images, len(dataset))
        indices = torch.randperm(len(dataset), generator=generator)[:count].tolist()
        first_sum = second_sum = None

        for start in range(0, count, 16):
            samples = [dataset[index] for index in indices[start:start + 16]]
            images = torch.stack([sample[0] for sample in samples]).to(device)
            labels = torch.tensor([int(sample[1]) for sample in samples], device=device)
            with torch.amp.autocast(device):
                losses = F.cross_entropy(logits_fn(images), labels, reduction="none")
            for index, loss in enumerate(losses):
                gradient = self._flat_gradient(loss, retain_graph=index + 1 < len(losses)).detach()
                first_sum = gradient.abs() if first_sum is None else first_sum + gradient.abs()
                second_sum = gradient.square() if second_sum is None else second_sum + gradient.square()

        self.first_moment = first_sum / count
        self.second_square = second_sum / count
        random.setstate(python_state)
        torch.random.set_rng_state(torch_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)

    def prepare(self, losses, step):
        if self.first_moment is None:
            raise RuntimeError("AbsIdentityGate.initialize must be called before training")
        selected = [
            ((step - 1) * self.online_images + index) % len(losses)
            for index in range(min(self.online_images, len(losses)))
        ]
        gradients = torch.stack([
            self._flat_gradient(losses[index], retain_graph=True).detach()
            for index in selected
        ])
        observed_first = gradients.abs().mean(dim=0)
        observed_second = gradients.square().mean(dim=0)
        self.first_moment.lerp_(observed_first, 1.0 - self.beta)
        self.second_square.lerp_(observed_second, 1.0 - self.beta)
        q = self.first_moment.square().div(self.second_square + self.epsilon).clamp_(0.0, 1.0)
        previous = [parameter.detach().clone() for parameter in self.parameters]
        return previous, q

    @torch.no_grad()
    def apply(self, previous, q):
        offset = 0
        for parameter, before in zip(self.parameters, previous):
            end = offset + parameter.numel()
            gate = q[offset:end].view_as(parameter).to(parameter.dtype)
            parameter.copy_(before + gate * (parameter - before))
            offset = end
