from typing import Union

from lightning.pytorch.callbacks import Callback
import torch
from torch.nn import Module


def grad_norm(module: Module, norm_type: Union[float, int, str], group_separator: str = "/") -> dict[str, float]:
    """Compute each parameter's gradient's norm and their overall norm.

    The overall norm is computed over all gradients together, as if they
    were concatenated into a single vector.

    Args:
        module: :class:`torch.nn.Module` to inspect.
        norm_type: The type of the used p-norm, cast to float if necessary.
            Can be ``'inf'`` for infinity norm.
        group_separator: The separator string used by the logger to group
            the gradients norms in their own subfolder instead of the logs one.

    Return:
        norms: The dictionary of p-norms of each parameter's gradient and
            a special entry for the total p-norm of the gradients viewed
            as a single vector.

    """
    norm_type = float(norm_type)
    if norm_type <= 0:
        raise ValueError(f"`norm_type` must be a positive number or 'inf' (infinity norm). Got {norm_type}")

    norms = {
        f"grad_{norm_type}_norm{group_separator}{name}": p.grad.data.norm(norm_type)
        for name, p in module.named_parameters()
        if p.grad is not None
    }
    if norms:
        total_norm = torch.tensor(list(norms.values())).norm(norm_type)
        norms[f"grad_{norm_type}_norm_total"] = total_norm
    return norms # type: ignore

class GradNormMonitor(Callback):
    def __init__(self, interval: int = 500):
        super().__init__()
        self.interval = interval

    def on_before_optimizer_step(self, trainer, pl_module, optimizer):
        if trainer.global_step % self.interval != 0:
            return

        gn = grad_norm(pl_module, norm_type=2)
        pl_module.log_dict({"grad/" + k: v for k, v in gn.items()}, prog_bar=False)