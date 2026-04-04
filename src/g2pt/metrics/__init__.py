from g2pt.metrics.l1 import L1Loss
from g2pt.metrics.mse import MSELoss
from g2pt.metrics.rl1 import RelL1Loss
from g2pt.metrics.rmse import RelMSELoss
from g2pt.metrics.rrmse import RootRelMSELoss
from g2pt.metrics.span import (
    BidiSpanLoss,
    GrassmannDistance,
    LstsqLoss,
    OrthogonalLoss,
    ProjectionLoss_NeurKItt,
    ProjectionLoss,
    SpanLoss,
    SelfDistance,
    InverseSpanLoss,
    PrincipalAngle
)


def get_metric(name: str, **kwargs):
    """
    Get a metric by name.

    Args:
        name (str): Name of the metric.
        **kwargs: Additional arguments for the metric.

    Returns:
        nn.Module: The metric class.
    """

    metrics = {
        "l1": L1Loss,
        "rl1": RelL1Loss,
        "mse": MSELoss,
        "rmse": RelMSELoss,
        "rrmse": RootRelMSELoss,
        "span": SpanLoss,
        "lstsq": LstsqLoss,
        "bidispan": BidiSpanLoss,
        "orthogonal": OrthogonalLoss,
        "inversespan": InverseSpanLoss,
        "projectfrob": ProjectionLoss,
        "grassmann": GrassmannDistance,
        "selfdistance": SelfDistance,
        "neurkitt": ProjectionLoss_NeurKItt,
        "principalangle": PrincipalAngle,
    }

    name = name.lower()  # Normalize the metric name to lowercase.
    name = name.replace(" ", "").replace("_", "")  # Remove spaces and underscores for consistency.
    if name not in metrics:
        raise ValueError(f"Metric {name} is not supported.")

    return metrics[name](**kwargs)  # Return an instance of the metric class.
