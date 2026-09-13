from datasets_loader.classification_dataset import CLASS_TO_INDEX
from utils.config import Config


def test_pe1_config_values():
    config = Config("configs/classification.yaml")

    assert config.get("seed") == 42
    assert config.get("dataset", "image_size") == 224
    assert config.get("training", "batch_size") == 32
    assert config.get("training", "epochs") == 50
    assert config.get("training", "learning_rate") == 1e-4
    assert config.get("training", "weight_decay") == 1e-5
    assert config.get("training", "validation_split") == 0.2
    assert config.get("model", "name") == "PE1Classifier"
    assert config.get("model", "backbone") == "efficientnet_b3"
    assert config.get("model", "num_classes") == 4
    assert config.get("optimizer", "name") == "Adam"
    assert config.get("scheduler", "name") == "ReduceLROnPlateau"
    assert config.get("scheduler", "factor") == 0.1
    assert config.get("scheduler", "patience") == 5
    assert config.get("loss", "name") == "CrossEntropyLoss"


def test_class_to_index_mapping():
    expected = {
        "glioma": 0,
        "meningioma": 1,
        "no_tumor": 2,
        "pituitary": 3,
    }
    assert CLASS_TO_INDEX == expected
