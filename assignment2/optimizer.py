from typing import Callable, Iterable, Tuple

import torch
from torch.optim import Optimizer
import math

class AdamW(Optimizer):
    def __init__(
            self,
            params: Iterable[torch.nn.parameter.Parameter],
            lr: float = 1e-3,
            betas: Tuple[float, float] = (0.9, 0.999),
            eps: float = 1e-6,
            weight_decay: float = 0.0,
            correct_bias: bool = True,
    ):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {} - should be >= 0.0".format(lr))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter: {} - should be in [0.0, 1.0[".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter: {} - should be in [0.0, 1.0[".format(betas[1]))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {} - should be >= 0.0".format(eps))
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, correct_bias=correct_bias)
        super().__init__(params, defaults)

    def step(self, closure: Callable = None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError("Adam does not support sparse gradients, please consider SparseAdam instead")

                # raise NotImplementedError()

                # State should be stored in this dictionary
                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    # Initialize steps
                    state['step'] = 0
                    # Initialize the first moments of the gradients
                    state["m"] = grad.new().resize_as_(grad).zero_()
                    # Initialize the second moments of the gradients
                    state["v"] = grad.new().resize_as_(grad).zero_()

                m, v = state["m"], state["v"]
                state["step"] += 1

                # Access hyperparameters from the `group` dictionary
                alpha = group["lr"]
                beta1, beta2 = group["betas"]
                correct_bias = group["correct_bias"]
                eps = group["eps"]
                weight_decay = group["weight_decay"]

                # Update first and second moments of the gradients
                m.mul_(beta1).add_(grad, alpha=1-beta1)
                v.mul_(beta2).addcmul_(grad, grad, value=1-beta2)

                # Bias correction
                # Please note that we are using the "efficient version" given in
                # https://arxiv.org/abs/1412.6980
                if correct_bias:
                    alpha = alpha * math.sqrt(1-beta2**state["step"]) / (1-beta1**state["step"])

                # Update parameters
                denom = v.sqrt().add_(eps)
                p.data.addcdiv_(m, denom, value=-alpha)
                # Add weight decay after the main gradient-based updates.
                # Please note that the learning rate should be incorporated into this update.
                p.data.add_(p.data, alpha=-weight_decay*group["lr"])
        return loss
