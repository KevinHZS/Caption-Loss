def combine_available_losses(loss_short=None, loss_long=None, long_loss_weight=1.0, short_loss_weight=1.0):
    weighted_loss_short = None if loss_short is None else short_loss_weight * loss_short
    weighted_loss_long = None if loss_long is None else long_loss_weight * loss_long
    if loss_short is not None and loss_long is not None:
        return weighted_loss_short + weighted_loss_long
    if weighted_loss_short is not None:
        return weighted_loss_short
    if weighted_loss_long is not None:
        return weighted_loss_long
    raise ValueError("At least one global contrastive loss must be available.")
