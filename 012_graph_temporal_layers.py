RUN_ABLATION = True

ABLATION_VARIANTS = ['Full', 'w/o CEEMDAN Branch', 'w/o Exogenous Branch', 'w/o Graph Convolution', 'w/o Gated Fusion', 'w/o Skip Connection', 'w/o Event Loss']

def graph_mm(A, X):
    if A.dim() == 2:
        return torch.einsum('ij,bjf->bif', A, X)
    elif A.dim() == 3:
        return torch.einsum('bij,bjf->bif', A, X)
    else:
        raise ValueError('A dim must be 2 or 3')

class ChebConv(nn.Module):

    def __init__(self, in_feats, out_feats):
        super().__init__()
        self.W0 = nn.Linear(in_feats, out_feats)
        self.W1 = nn.Linear(in_feats, out_feats)
        self.W2 = nn.Linear(in_feats, out_feats)

    def forward(self, X, A):
        T0 = X
        T1 = graph_mm(A, X)
        T2 = 2.0 * graph_mm(A, T1) - X
        return self.W0(T0) + self.W1(T1) + self.W2(T2)

class GatedTemporalBlock(nn.Module):

    def __init__(self, channels, kernel_size=3, dropout=0.1):
        super().__init__()
        self.filter_conv = nn.Conv1d(channels, channels, kernel_size=kernel_size, padding=kernel_size // 2)
        self.gate_conv = nn.Conv1d(channels, channels, kernel_size=kernel_size, padding=kernel_size // 2)
        self.norm = nn.BatchNorm1d(channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        filt = torch.tanh(self.filter_conv(x))
        gate = torch.sigmoid(self.gate_conv(x))
        out = filt * gate
        out = self.norm(out)
        out = self.dropout(out)
        return out + x

class BranchEncoder(nn.Module):

    def __init__(self, in_feats, hidden, hist, dropout=0.1, use_graph=True):
        super().__init__()
        self.hist = hist
        self.hidden = hidden
        self.use_graph = use_graph
        self.in_proj = nn.Linear(in_feats, hidden)
        if self.use_graph:
            self.gconv1 = ChebConv(hidden, hidden)
            self.gconv2 = ChebConv(hidden, hidden)
        else:
            self.ffn1 = nn.Linear(hidden, hidden)
            self.ffn2 = nn.Linear(hidden, hidden)
        self.temporal1 = GatedTemporalBlock(hidden, kernel_size=3, dropout=dropout)
        self.temporal2 = GatedTemporalBlock(hidden, kernel_size=3, dropout=dropout)
        self.attn_fc = nn.Linear(hidden, max(16, hidden // 2))
        self.attn_score = nn.Linear(max(16, hidden // 2), 1)
        self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(hidden)

    def forward(self, X_seq, A_seq):
        B, T, N, _ = X_seq.shape
        hs = []
        for t in range(T):
            xt = X_seq[:, t, :, :]
            At = A_seq[:, t, :, :]
            h = self.in_proj(xt)
            if self.use_graph:
                h = F.gelu(self.gconv1(h, At))
                h = self.dropout(h)
                h = F.gelu(self.gconv2(h, At))
                h = self.dropout(h)
            else:
                h = F.gelu(self.ffn1(h))
                h = self.dropout(h)
                h = F.gelu(self.ffn2(h))
                h = self.dropout(h)
            hs.append(h)
        H = torch.stack(hs, dim=1)
        H = H.permute(0, 2, 3, 1).contiguous().view(B * N, self.hidden, T)
        H = self.temporal1(H)
        H = self.temporal2(H)
        H = H.view(B, N, self.hidden, T).permute(0, 3, 1, 2).contiguous()
        H = self.ln(H)
        attn_logits = self.attn_score(torch.tanh(self.attn_fc(H))).squeeze(-1)
        attn = torch.softmax(attn_logits, dim=1).unsqueeze(-1)
        H_attn = (H * attn).sum(dim=1)
        H_last = H[:, -1, :, :]
        out = 0.6 * H_last + 0.4 * H_attn
        return out
