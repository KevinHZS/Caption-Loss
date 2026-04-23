def combine_available_losses(loss_short=None, loss_long=None):
    if loss_short is not None and loss_long is not None:
        return loss_short + loss_long
    if loss_short is not None:
        return loss_short
    if loss_long is not None:
        return loss_long
    raise ValueError("At least one global contrastive loss must be available.")
