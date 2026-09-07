import torch
from torchmetrics.classification import Accuracy
from tqdm import tqdm
from itertools import zip_longest

from src.evaluation.metrics import AverageMeter


def train_knowledge_distillation(
    student,
    teacher,
    student_loader,
    teacher_loader,
    kd_loss_fn,
    optimizer,
    device,
    epoch=None,
    num_classes=5,
):
    """
    Train the student model using knowledge distillation.

    The teacher model is used only to generate soft targets.
    Teacher parameters are not updated during student training.
    """

    student.train()

    loss_train = AverageMeter()

    acc_train = Accuracy(
        task="multiclass",
        num_classes=num_classes
    ).to(device)

    total_batches = max(
        len(student_loader),
        len(teacher_loader)
    )

    progress_bar = tqdm(
        total=total_batches,
        unit="batch"
    )

    if epoch is not None:
        progress_bar.set_description(
            f"Epoch {epoch}"
        )

    for (inputs_student, targets_student), (
        inputs_teacher,
        targets_teacher
    ) in zip_longest(
        student_loader,
        teacher_loader,
        fillvalue=(None, None)
    ):

        # Move data to device
        inputs_student = inputs_student.to(device)
        targets_student = targets_student.to(device)

        inputs_teacher = inputs_teacher.to(device)

        # Student prediction
        student_outputs = student(
            inputs_student
        )

        # Teacher prediction
        with torch.no_grad():

            teacher_outputs = teacher(
                inputs_teacher
            )

        # Knowledge distillation loss
        loss = kd_loss_fn(
            student_outputs,
            targets_student,
            teacher_outputs,
            temperature=2.0,
            alpha=0.4,
        )

        # Backpropagation
        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        # Update statistics
        loss_train.update(
            loss.item()
        )

        acc_train.update(
            student_outputs,
            targets_student.int()
        )

        progress_bar.set_postfix(
            loss=loss_train.avg,
            accuracy=100.0 * acc_train.compute().item(),
        )

        progress_bar.update(1)

    progress_bar.close()

    return (
        student,
        loss_train.avg,
        acc_train.compute().item(),
    )
