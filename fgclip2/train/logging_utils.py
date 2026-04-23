def prepare_loss_metrics_for_logging(loss_dict):
    metrics = {}
    if not loss_dict:
        return metrics

    for name, value in loss_dict.items():
        if value is None:
            continue
        if hasattr(value, "detach"):
            value = value.detach()
        if hasattr(value, "item"):
            value = value.item()
        metrics[name] = float(value)
    return metrics
