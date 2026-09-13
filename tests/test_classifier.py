import torch

from models.pe1_classifier import PE1Classifier


def test_pe1_classifier_forward_shape():
    model = PE1Classifier(
        pretrained=False,
    )

    dummy = torch.randn(
        2,
        3,
        224,
        224,
    )

    output = model(dummy)

    assert output.shape == (
        2,
        4,
    )


def test_pe1_parameter_count():
    model = PE1Classifier(
        pretrained=False,
    )

    total_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    assert total_parameters == 11488300
