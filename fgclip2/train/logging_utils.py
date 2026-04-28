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


def _accumulate_grad_norm_sq(total, param):
    if param.grad is None or not param.requires_grad:
        return total
    grad = param.grad.detach()
    return total + float(grad.float().norm(2).item() ** 2)


def compute_module_grad_norms(model):
    norm_squares = {
        "grad_norm_projector": 0.0,
        "grad_norm_vision": 0.0,
        "grad_norm_text": 0.0,
        "grad_norm_total_trainable": 0.0,
    }
    has_grad = {name: False for name in norm_squares}

    for name, param in model.named_parameters():
        if param.grad is None or not param.requires_grad:
            continue

        norm_squares["grad_norm_total_trainable"] = _accumulate_grad_norm_sq(
            norm_squares["grad_norm_total_trainable"], param
        )
        has_grad["grad_norm_total_trainable"] = True

        if "llm_caption_decoder.projector" in name:
            norm_squares["grad_norm_projector"] = _accumulate_grad_norm_sq(norm_squares["grad_norm_projector"], param)
            has_grad["grad_norm_projector"] = True
        elif "vision_model" in name:
            norm_squares["grad_norm_vision"] = _accumulate_grad_norm_sq(norm_squares["grad_norm_vision"], param)
            has_grad["grad_norm_vision"] = True
        elif any(key in name for key in ("text_model", "longtext_head", "boxtext_head")):
            norm_squares["grad_norm_text"] = _accumulate_grad_norm_sq(norm_squares["grad_norm_text"], param)
            has_grad["grad_norm_text"] = True

    return {name: value**0.5 for name, value in norm_squares.items() if has_grad[name]}
