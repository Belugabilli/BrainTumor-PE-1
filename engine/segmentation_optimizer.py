import torch


def create_segmentation_optimizer(
    model,
    learning_rate=1e-4,
    weight_decay=1e-5,
):
    """
    Create Adam optimizer using only trainable parameters.
    """

    trainable_parameters = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad
    ]

    optimizer = torch.optim.Adam(
        trainable_parameters,
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    return optimizer


def create_segmentation_scheduler(
    optimizer,
    T_0=10,
    T_mult=2,
    eta_min=1e-6,
):
    """
    CosineAnnealingWarmRestarts scheduler.
    """

    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=T_0,
        T_mult=T_mult,
        eta_min=eta_min,
    )

    return scheduler