from datasets_loader.classification_dataset import (
    BrainTumorClassificationDataset,
)

from datasets_loader.transforms import (
    get_train_transforms,
)

def test_classification_dataset_loads_pe1_train_split():
    dataset = BrainTumorClassificationDataset(
        root_dir="datasets/classification_task/train",
        transform=get_train_transforms(),
    )

    assert len(dataset) == 5000

    image, label = dataset[0]

    assert image.shape == (
        3,
        224,
        224,
    )
    assert label in {
        0,
        1,
        2,
        3,
    }
