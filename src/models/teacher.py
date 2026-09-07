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


class TeacherModel(nn.Module):
    """
    T-CST: Teacher model based on CNN, SENet, and Transformer.

    The model extracts local temporal features using a deep CNN-SENet
    backbone and captures long-range temporal dependencies using a
    Transformer encoder.

    Parameters
    ----------
    input_size : int
        Number of input ECG leads/channels.

    cnn_hidden_size : int
        Hidden dimension used by the Transformer encoder.

    nhead : int
        Number of attention heads.

    num_enc : int
        Number of Transformer encoder layers.

    dim_feedforward : int
        Dimension of the Transformer feed-forward network.

    dropout : float
        Dropout probability.

    activation : str or callable
        Activation function used in the Transformer.

    num_cls : int
        Number of output classes.
    """

    def __init__(
        self,
        input_size: int,
        cnn_hidden_size: int,
        nhead: int,
        num_enc: int,
        dim_feedforward: int,
        dropout: float,
        activation,
        num_cls: int = 5,
    ):
        super().__init__()

        # ---------------------------------------------------------
        # CNN backbone
        # ---------------------------------------------------------

        self.conv1 = nn.Conv1d(input_size, 32, kernel_size=5, padding=1)
        self.bn1 = nn.BatchNorm1d(32)
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(32, 32, kernel_size=5, padding=1)
        self.bn2 = nn.BatchNorm1d(32)
        self.dropout2 = nn.Dropout(dropout)

        self.conv11 = nn.Conv1d(32, 32, kernel_size=5, padding=1)
        self.bn11 = nn.BatchNorm1d(32)
        self.dropout11 = nn.Dropout(dropout)

        self.conv33 = nn.Conv1d(32, 32, kernel_size=5, padding=1)
        self.bn33 = nn.BatchNorm1d(32)
        self.dropout33 = nn.Dropout(dropout)

        self.conv3 = nn.Conv1d(32, 64, kernel_size=5, padding=1)
        self.bn3 = nn.BatchNorm1d(64)
        self.dropout3 = nn.Dropout(dropout)

        self.conv4 = nn.Conv1d(64, 64, kernel_size=5, padding=1)
        self.bn4 = nn.BatchNorm1d(64)
        self.dropout4 = nn.Dropout(dropout)

        self.conv12 = nn.Conv1d(64, 64, kernel_size=5, padding=1)
        self.bn12 = nn.BatchNorm1d(64)
        self.dropout12 = nn.Dropout(dropout)

        self.conv22 = nn.Conv1d(64, 64, kernel_size=5, padding=1)
        self.bn22 = nn.BatchNorm1d(64)
        self.dropout22 = nn.Dropout(dropout)

        self.conv5 = nn.Conv1d(64, 128, kernel_size=5, padding=1)
        self.bn5 = nn.BatchNorm1d(128)
        self.dropout5 = nn.Dropout(dropout)

        self.conv6 = nn.Conv1d(128, 128, kernel_size=5, padding=1)
        self.bn6 = nn.BatchNorm1d(128)
        self.dropout6 = nn.Dropout(dropout)

        self.conv7 = nn.Conv1d(128, 256, kernel_size=5, padding=1)
        self.bn7 = nn.BatchNorm1d(256)
        self.dropout7 = nn.Dropout(dropout)

        self.conv8 = nn.Conv1d(256, 256, kernel_size=5, padding=1)
        self.bn8 = nn.BatchNorm1d(256)
        self.dropout8 = nn.Dropout(dropout)

        self.conv9 = nn.Conv1d(256, 512, kernel_size=5, padding=1)
        self.bn9 = nn.BatchNorm1d(512)
        self.dropout9 = nn.Dropout(dropout)

        self.conv10 = nn.Conv1d(512, 512, kernel_size=5, padding=1)
        self.bn10 = nn.BatchNorm1d(512)
        self.dropout10 = nn.Dropout(dropout)

        # ---------------------------------------------------------
        # Squeeze-and-Excitation blocks
        # ---------------------------------------------------------

        self.se1 = SELayer(channel=32)
        self.se2 = SELayer(channel=32)
        self.se3 = SELayer(channel=32)
        self.se33 = SELayer(channel=32)

        self.se4 = SELayer(channel=64)
        self.se44 = SELayer(channel=64)
        self.se444 = SELayer(channel=64)
        self.se22 = SELayer(channel=64)

        self.se5 = SELayer(channel=128)
        self.se6 = SELayer(channel=128)

        self.se7 = SELayer(channel=256)
        self.se8 = SELayer(channel=256)

        self.se9 = SELayer(channel=512)
        self.se10 = SELayer(channel=512)

        # ---------------------------------------------------------
        # Transformer encoder
        # ---------------------------------------------------------

        self.encoder = nn.Transformer(
            d_model=cnn_hidden_size,
            nhead=nhead,
            num_encoder_layers=num_enc,
            num_decoder_layers=0,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            batch_first=True,
        ).encoder

        # ---------------------------------------------------------
        # Fully connected layers
        # ---------------------------------------------------------

        self.layer0 = nn.LazyLinear(1000)
        self.bn0 = nn.LazyBatchNorm1d()
        self.dropout0 = nn.Dropout(dropout)

        self.layer03 = nn.LazyLinear(360)
        self.bn03 = nn.LazyBatchNorm1d()
        self.dropout03 = nn.Dropout(dropout)

        self.layer02 = nn.LazyLinear(360)
        self.bn02 = nn.LazyBatchNorm1d()
        self.dropout02 = nn.Dropout(dropout)

        self.layer01 = nn.LazyLinear(720)
        self.bn01 = nn.LazyBatchNorm1d()
        self.dropout01 = nn.Dropout(dropout)

        self.fc = nn.LazyLinear(num_cls)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Expected input shape:
            (batch_size, sequence_length, input_size)

        Returns:
            Classification logits with shape:
            (batch_size, num_cls)
        """

        # Convert to Conv1D format:
        # (B, L, C) -> (B, C, L)
        x = x.permute(0, 2, 1)

        # ---------------------------------------------------------
        # CNN + SENet feature extraction
        # ---------------------------------------------------------

        y = self.dropout1(self.bn1(self.conv1(x)).relu())
        y = self.se1(y)

        y = self.dropout2(self.bn2(self.conv2(y)).relu())
        y = self.se2(y)

        y = self.dropout11(self.bn11(self.conv11(y)).relu())
        y = self.se3(y)

        y = self.dropout33(self.bn33(self.conv33(y)).relu())
        y = self.se33(y)

        y = self.dropout3(self.bn3(self.conv3(y)).relu())
        y = self.se4(y)

        y = self.dropout4(self.bn4(self.conv4(y)).relu())
        y = self.se44(y)

        y = self.dropout12(self.bn12(self.conv12(y)).relu())
        y = self.se444(y)

        y = self.dropout5(self.bn5(self.conv5(y)).relu())
        y = self.se5(y)

        y = self.dropout6(self.bn6(self.conv6(y)).relu())
        y = self.se6(y)

        y = self.dropout7(self.bn7(self.conv7(y)).relu())
        y = self.se7(y)

        y = self.dropout8(self.bn8(self.conv8(y)).relu())
        y = self.se8(y)

        y = self.dropout9(self.bn9(self.conv9(y)).relu())
        y = self.se9(y)

        y = self.dropout10(self.bn10(self.conv10(y)).relu())
        y = self.se10(y)

        # ---------------------------------------------------------
        # Transformer
        # ---------------------------------------------------------

        # (B, C, L) -> (B, L, C)
        y = y.permute(0, 2, 1)

        y = self.dropout0(self.bn0(self.layer0(y)).relu())
        y = self.dropout03(self.bn03(self.layer03(y)).relu())

        y = self.encoder(y)

        y = self.dropout01(self.bn01(self.layer01(y)).relu())
        y = self.dropout02(self.bn02(self.layer02(y)).relu())

        # Use the final temporal representation for classification
        y = self.fc(y[:, -1])

        return y
