import torch.nn.functional as F


def knowledge_distillation_loss(
    student_outputs,
    labels,
    teacher_outputs,
    temperature=2.0,
    alpha=0.3,
):
    """
    Response-based Knowledge Distillation loss.

    Combines the distillation loss based on KL divergence
    with the standard cross-entropy classification loss.
    """

    distillation_loss = F.kl_div(
        F.log_softmax(student_outputs / temperature, dim=1),
        F.softmax(teacher_outputs / temperature, dim=1),
        reduction="batchmean",
    ) * (alpha * temperature**2)

    classification_loss = F.cross_entropy(
        student_outputs,
        labels,
    ) * (1.0 - alpha)

    return distillation_loss + classification_loss
