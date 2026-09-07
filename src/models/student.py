import torch
import torch.nn as nn


class SELayer(nn.Module):
    """
    Squeeze-and-Excitation (SE) block for 1D feature maps.
    """

    def __init__(self, channel: int, reduction: int = 16):
        super().__init__()

        self.avg_pool = nn.AdaptiveAvgPool1d(1)

        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, channels, _ = x.size()

        y = self.avg_pool(x).view(batch_size, channels)
        y = self.fc(y).view(batch_size, channels, 1)

        return x * y.expand_as(x)


class StudentModel(nn.Module):
    """
    S-CS: Lightweight student model based on CNN and SENet.

    The model is designed as the lightweight counterpart of the
    T-CST teacher model for efficient ECG classification.

    Parameters
    ----------
    input_size : int
        Number of input ECG leads/channels.

    dropout : float
        Dropout probability.

    activation : str or callable
        Activation function parameter retained for compatibility
        with the original implementation.

    num_cls : int
        Number of output classes.
    """

    def __init__(
        self,
        input_size: int,
        dropout: float,
        activation,
        num_cls: int = 5,
    ):
        super().__init__()

        # ---------------------------------------------------------
        # CNN backbone
        # ---------------------------------------------------------

        self.conv1 = nn.Conv1d(
            input_size, 16, kernel_size=15, padding=1
        )
        self.bn1 = nn.BatchNorm1d(16)
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(
            16, 16, kernel_size=15, padding=1
        )
        self.bn2 = nn.BatchNorm1d(16)
        self.dropout2 = nn.Dropout(dropout)

        self.conv11 = nn.Conv1d(
            16, 16, kernel_size=15, padding=1
        )
        self.bn11 = nn.BatchNorm1d(16)
        self.dropout11 = nn.Dropout(dropout)

        self.conv3 = nn.Conv1d(
            16, 16, kernel_size=15, padding=1
        )
        self.bn3 = nn.BatchNorm1d(16)
        self.dropout3 = nn.Dropout(dropout)

        self.conv4 = nn.Conv1d(
            16, 32, kernel_size=15, padding=1
        )
        self.bn4 = nn.BatchNorm1d(32)
        self.dropout4 = nn.Dropout(dropout)

        self.conv12 = nn.Conv1d(
            32, 32, kernel_size=15, padding=1
        )
        self.bn12 = nn.BatchNorm1d(32)
        self.dropout12 = nn.Dropout(dropout)

        self.conv5 = nn.Conv1d(
            32, 32, kernel_size=15, padding=1
        )
        self.bn5 = nn.BatchNorm1d(32)
        self.dropout5 = nn.Dropout(dropout)

        self.conv6 = nn.Conv1d(
            32, 32, kernel_size=15, padding=1
        )
        self.bn6 = nn.BatchNorm1d(32)
        self.dropout6 = nn.Dropout(dropout)

        self.conv7 = nn.Conv1d(
            32, 64, kernel_size=15, padding=1
        )
        self.bn7 = nn.BatchNorm1d(64)
        self.dropout7 = nn.Dropout(dropout)

        self.conv8 = nn.Conv1d(
            64, 64, kernel_size=15, padding=1
        )
        self.bn8 = nn.BatchNorm1d(64)
        self.dropout8 = nn.Dropout(dropout)

        # ---------------------------------------------------------
        # Squeeze-and-Excitation blocks
        # ---------------------------------------------------------

        self.se1 = SELayer(channel=16)
        self.se2 = SELayer(channel=16)
        self.se3 = SELayer(channel=16)
        self.se4 = SELayer(channel=16)

        self.se44 = SELayer(channel=32)
        self.se444 = SELayer(channel=32)

        self.se5 = SELayer(channel=32)
        self.se6 = SELayer(channel=32)

        self.se7 = SELayer(channel=64)
        self.se8 = SELayer(channel=64)

        # ---------------------------------------------------------
        # Fully connected layers
        # ---------------------------------------------------------

        self.layer0 = nn.LazyLinear(64)
        self.bn0 = nn.LazyBatchNorm1d()
        self.dropout0 = nn.Dropout(dropout)

        self.layer03 = nn.LazyLinear(16)
        self.bn03 = nn.LazyBatchNorm1d()
        self.dropout03 = nn.Dropout(dropout)

        self.fc = nn.LazyLinear(num_cls)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Expected input shape:
            (batch_size, sequence_length, input_size)

        Returns
        -------
        torch.Tensor
            Classification logits with shape:
            (batch_size, num_cls)
        """

        # Convert to Conv1D format:
        # (B, L, C) -> (B, C, L)
        x = x.permute(0, 2, 1)

        # ---------------------------------------------------------
        # CNN + SENet feature extraction
        # ---------------------------------------------------------

        y = self.dropout1(
            self.bn1(self.conv1(x)).relu()
        )
        y = self.se1(y)

        y = self.dropout2(
            self.bn2(self.conv2(y)).relu()
        )
        y = self.se2(y)

        y = self.dropout11(
            self.bn11(self.conv11(y)).relu()
        )
        y = self.se3(y)

        y = self.dropout3(
            self.bn3(self.conv3(y)).relu()
        )
        y = self.se4(y)

        y = self.dropout4(
            self.bn4(self.conv4(y)).relu()
        )
        y = self.se44(y)

        y = self.dropout12(
            self.bn12(self.conv12(y)).relu()
        )
        y = self.se444(y)

        y = self.dropout5(
            self.bn5(self.conv5(y)).relu()
        )
        y = self.se5(y)

        y = self.dropout6(
            self.bn6(self.conv6(y)).relu()
        )
        y = self.se6(y)

        y = self.dropout7(
            self.bn7(self.conv7(y)).relu()
        )
        y = self.se7(y)

        y = self.dropout8(
            self.bn8(self.conv8(y)).relu()
        )
        y = self.se8(y)

        # ---------------------------------------------------------
        # Fully connected classification head
        # ---------------------------------------------------------

        # (B, C, L) -> (B, L, C)
        y = y.permute(0, 2, 1)

        y = self.dropout0(
            self.bn0(self.layer0(y)).relu()
        )

        y = self.dropout03(
            self.bn03(self.layer03(y)).relu()
        )

        # Flatten temporal and feature dimensions
        y = torch.flatten(y, 1)

        # Classification logits
        y = self.fc(y)

        return y
