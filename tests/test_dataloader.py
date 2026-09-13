from utils.config import Config
from datasets_loader.create_dataloaders import create_dataloaders


def test_dataloaders_split():
    config = Config("configs/classification.yaml")
    config.config["dataset"]["num_workers"] = 0
    train_loader, val_loader = create_dataloaders(config)


    assert len(train_loader.dataset) == 4000
    assert len(val_loader.dataset) == 1000

    images, labels = next(iter(train_loader))
    assert images.shape[1:] == (3, 224, 224)
    assert len(labels) == config.get("training", "batch_size")