from .UNET_trainer import UNET2D_trainer
from .UNET3D_trainer import UNET3D_trainer


def load_trainer(model_name, loaders, logger, args):
    assert model_name in ("UNET", "UNET3D")

    if model_name == "UNET":
        return UNET2D_trainer(loaders, logger, args)
    elif model_name == "UNET3D":
        return UNET3D_trainer(loaders, logger, args)
    else:
        raise NotImplementedError("This trainer has not been implemented yet")
