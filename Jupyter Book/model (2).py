import torch
import torch.nn as nn

class SignLanguageModel(nn.Module):
    def __init__(self, input_size=258, hidden_size=128, num_classes=100):
        super().__init__()

        # ================= FEATURE REFINEMENT =================
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
        )

        # ================= TEMPORAL MODEL =================
        self.lstm = nn.LSTM(
            input_size=256,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.1
        )

        # ================= ATTENTION POOLING =================
        self.attention = nn.Sequential(
            nn.Linear(hidden_size * 2, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )

        # ================= CLASSIFIER =================
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        # x: (batch, frames, features)

        B, T, F = x.shape

        # ===== Feature extraction per frame =====
        x = x.view(B * T, F)
        x = self.feature_extractor(x)
        x = x.view(B, T, -1)

        # ===== LSTM =====
        lstm_out, _ = self.lstm(x)   # (B, T, H*2)

        # ===== Attention pooling =====
        attn_weights = self.attention(lstm_out)  # (B, T, 1)
        attn_weights = torch.softmax(attn_weights, dim=1)

        context = torch.sum(attn_weights * lstm_out, dim=1)

        # ===== Classification =====
        out = self.classifier(context)

        return out


# ====================== QUICK TEST ======================
if __name__ == "__main__":
    model = SignLanguageModel(num_classes=100)
    dummy_input = torch.randn(8, 30, 258)
    output = model(dummy_input)
    
    print(f"Input Shape : {dummy_input.shape}")   # [8, 30, 258]
    print(f"Output Shape: {output.shape}")        # [8, 100]
    print("Model test passed! ✅")