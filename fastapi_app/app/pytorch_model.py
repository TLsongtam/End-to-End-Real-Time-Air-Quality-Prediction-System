import torch
import torch.nn as nn
import torch.nn.functional as F

# ============================================================
# 1. LOCAL ATTENTION MECHANISM (LAM)
# ============================================================
class LocalAttention(nn.Module):
    def __init__(self, hidden_size, window_size=3):
        super().__init__()
        self.window_size = window_size

        self.W_q = nn.Linear(hidden_size, hidden_size, bias=False)
        self.W_k = nn.Linear(hidden_size, hidden_size, bias=False)
        self.W_v = nn.Linear(hidden_size, hidden_size, bias=False)

        self.scale = hidden_size ** 0.5
        self.out_proj = nn.Linear(hidden_size, hidden_size)

    def forward(self, x):
        B, T, H = x.size()
        w = self.window_size

        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)

        pad = w // 2
        K_pad = F.pad(K.permute(0, 2, 1), (pad, pad), mode='replicate').permute(0, 2, 1)
        V_pad = F.pad(V.permute(0, 2, 1), (pad, pad), mode='replicate').permute(0, 2, 1)

        out = []
        for t in range(T):
            k_local = K_pad[:, t : t + w, :]
            v_local = V_pad[:, t : t + w, :]
            q_t     = Q[:, t, :].unsqueeze(1)

            score = torch.bmm(q_t, k_local.transpose(1, 2)) / self.scale
            attn  = F.softmax(score, dim=-1)

            context = torch.bmm(attn, v_local)
            out.append(context)

        out = torch.cat(out, dim=1)
        return self.out_proj(out)

# ============================================================
# 2. MULTI-SCALE LOCAL ATTENTION MODULE (MLAM)
# ============================================================
class MultiScaleBlock(nn.Module):
    def __init__(self, hidden_size, window_sizes=(3, 6, 12)):
        super().__init__()
        self.attentions = nn.ModuleList([
            LocalAttention(hidden_size, ws) for ws in window_sizes
        ])
        self.fusion = nn.Linear(hidden_size * len(window_sizes), hidden_size)
        self.norm   = nn.LayerNorm(hidden_size)

    def forward(self, x):
        outs = [attn(x) for attn in self.attentions]
        concat = torch.cat(outs, dim=-1)
        fused  = self.fusion(concat)
        return self.norm(fused + x) # Residual connection

# ============================================================
# 3. BiLSTM-MLAM MODEL (Đã sửa theo phản biện)
# ============================================================
class BiLSTM_MLAM(nn.Module):
    """
    Kiến trúc chuẩn theo sơ đồ và phản biện:
    Input (X) ──> Stacked BiLSTM (2 Layers) ──> Khối MLAM (3 Scales) ──> FC Layer ──> Output
    """
    def __init__(
        self,
        input_size,
        hidden_size  = 64,
        output_size  = 24,
        num_layers   = 2,           # SỬA ĐÚNG: 2 lớp BiLSTM xếp chồng theo chiều dọc
        window_sizes = (3, 6, 12),
        dropout      = 0.2,
    ):
        super().__init__()
        
        # Lớp Stacked BiLSTM hoàn chỉnh
        self.bilstm = nn.LSTM(
            input_size    = input_size,
            hidden_size   = hidden_size,
            num_layers    = num_layers,  # = 2 
            batch_first   = True,
            bidirectional = True,
            dropout       = dropout,     # THÊM ĐÚNG: Dropout giữa layer 1 và layer 2 của LSTM
        )
        
        bilstm_output_dim = hidden_size * 2 

        # Khối MLAM nhận đầu vào từ tầng cuối của Stacked BiLSTM
        self.mlam = MultiScaleBlock(bilstm_output_dim, window_sizes)

        # Lớp đầu ra cuối cùng
        self.dropout = nn.Dropout(dropout)
        self.fc      = nn.Linear(bilstm_output_dim, output_size)

    def forward(self, x):
        # 1. Trích xuất đặc trưng qua 2 lớp BiLSTM xếp chồng
        bilstm_out, _ = self.bilstm(x)            # (B, T, hidden_size * 2)

        # 2. Đi qua khối MLAM (3 nhánh song song -> Feature Fusion)
        mlam_out = self.mlam(bilstm_out)          # (B, T, hidden_size * 2)

        # 3. Lấy timestep cuối cùng và đưa qua Fully Connected
        feat_vector = mlam_out[:, -1, :]          # (B, hidden_size * 2)
        feat_vector = self.dropout(feat_vector)
        output = self.fc(feat_vector)             # (B, output_size)
        
        return output

# ============================================================
# 4. KHỞI TẠO VÀ KIỂM TRA (DUMMY TEST)
# ============================================================
def build_model(input_size, config='48_24', station_id=None):
    model = BiLSTM_MLAM(
        input_size   = input_size,
        hidden_size  = 64,
        output_size  = 24,
        num_layers   = 2, # Khởi tạo mặc định 2 layers
        window_sizes = (3, 6, 12),
        dropout      = 0.2,
    )
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    label = f"Trạm {station_id}" if station_id else ""
    print(f"  [{config}] {label} | input_size={input_size} | params={total_params:,}")
    return model
