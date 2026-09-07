import torch
from torchmetrics.classification import Accuracy

from .metrics import AverageMeter


def validate(model, test_loader, loss_fn, device, num_classes=5):
    """
    Evaluate the model on the validation dataset.

    Returns
    -------
    tuple
        Average validation loss and accuracy.
    """

    model.eval()

    loss_valid = AverageMeter()
    acc_valid = Accuracy(
        task="multiclass",
        num_classes=num_classes
    ).to(device)

    with torch.no_grad():

        for inputs, targets in test_loader:

            inputs = inputs.to(device)
            targets = targets.to(device)

            outputs = model(inputs)

            loss = loss_fn(outputs, targets)

            loss_valid.update(loss.item())

            acc_valid.update(
                outputs,
                targets.int()
            )

    return (
        loss_valid.avg,
        acc_valid.compute().item()
    )


def test(model, test_loader, loss_fn, device, num_classes=5):
    """
    Evaluate the trained model on the test dataset.

    Returns
    -------
    tuple
        Average test loss and accuracy.
    """

    model.eval()

    loss_test = AverageMeter()
    acc_test = Accuracy(
        task="multiclass",
        num_classes=num_classes
    ).to(device)

    with torch.no_grad():

        for inputs, targets in test_loader:

            inputs = inputs.to(device)
            targets = targets.to(device)

            outputs = model(inputs)

            loss = loss_fn(outputs, targets)

            loss_test.update(loss.item())

            acc_test.update(
                outputs,
                targets.int()
            )

    return (
        loss_test.avg,
        acc_test.compute().item()
    )
