import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalAttention(nn.Module):
    """
    Soft attention over the LSTM time steps.
    Lets the model focus on the most informative frames
    instead of blindly using the last timestep.
    """
    def __init__(self, hidden_size):
        super().__init__()
        self.attn = nn.Linear(hidden_size, 1)

    def forward(self, lstm_out):
        # lstm_out: (B, T, H)
        scores  = self.attn(lstm_out).squeeze(-1)             # (B, T)
        weights = torch.softmax(scores, dim=1)                # (B, T)
        context = (lstm_out * weights.unsqueeze(-1)).sum(dim=1)  # (B, H)
        return context


class SignLanguageModel(nn.Module):
    def __init__(self, input_size=300, num_classes=100, dropout=0.35):
        super().__init__()

        # ------- 1D-CNN: local motion feature extractor -------
        # Wider now that overfitting gap is only 2% — model can handle more capacity
        self.conv1d = nn.Sequential(
            nn.Conv1d(input_size, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(dropout),

            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.GELU(),

            # Extra conv layer — learns higher-level motion patterns
            nn.Conv1d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.MaxPool1d(kernel_size=2),   # T -> T/2
        )

        # ------- Bi-LSTM: temporal sequence modeling -------
        self.lstm = nn.LSTM(
            input_size=256,
            hidden_size=128,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=dropout
        )

        lstm_out_size = 128 * 2  # bidirectional -> 256

        # ------- Attention over time steps -------
        self.attention = TemporalAttention(lstm_out_size)

        # ------- Classification head -------
        # attention + mean pool + max pool concatenated -> 768
        self.classifier = nn.Sequential(
            nn.Linear(lstm_out_size * 3, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(dropout / 2),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        # x: (B, T, F)  e.g. (32, 30, 300)

        # CNN expects (B, F, T)
        x = x.transpose(1, 2)
        x = self.conv1d(x)            # (B, 256, T/2)

        # LSTM expects (B, T, F)
        x = x.transpose(1, 2)         # (B, T/2, 256)
        lstm_out, _ = self.lstm(x)     # (B, T/2, 256)

        # Three complementary pooling strategies
        attn_out  = self.attention(lstm_out)       # (B, 256) — focus on key frames
        mean_pool = lstm_out.mean(dim=1)           # (B, 256) — overall motion summary
        max_pool  = lstm_out.max(dim=1).values     # (B, 256) — peak activation

        combined = torch.cat([attn_out, mean_pool, max_pool], dim=-1)  # (B, 768)

        return self.classifier(combined)