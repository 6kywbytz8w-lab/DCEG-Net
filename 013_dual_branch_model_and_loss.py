class DualBranchCEGNet(nn.Module):

    def __init__(self, n_nodes, comp_dim, exo_dim, hidden, hist, skip_idx_list=None, dropout=0.1, use_graph=True, use_gated_fusion=True, use_skip=True):
        super().__init__()
        self.n_nodes = n_nodes
        self.hist = hist
        self.hidden = hidden
        self.skip_idx_list = skip_idx_list if skip_idx_list is not None else []
        self.use_gated_fusion = use_gated_fusion
        self.use_skip = use_skip
        self.comp_branch = BranchEncoder(comp_dim, hidden, hist, dropout=dropout, use_graph=use_graph)
        self.exo_branch = BranchEncoder(exo_dim, hidden, hist, dropout=dropout, use_graph=use_graph)
        self.comp_gate = nn.Sequential(nn.Linear(hidden, hidden), nn.Sigmoid())
        self.exo_gate = nn.Sequential(nn.Linear(hidden, hidden), nn.Sigmoid())
        self.fuse_gate = nn.Sequential(nn.Linear(hidden * 2, hidden), nn.GELU(), nn.Linear(hidden, hidden), nn.Sigmoid())
        if self.use_skip and len(self.skip_idx_list) > 0:
            self.skip_proj = nn.Linear(len(self.skip_idx_list), hidden // 4)
            skip_dim = hidden // 4
        else:
            self.skip_proj = None
            skip_dim = 0
        self.main_head = nn.Sequential(nn.Linear(hidden + skip_dim, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, 1))
        self.event_head = nn.Sequential(nn.Linear(hidden + skip_dim, hidden // 2), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden // 2, 1))
        self.calib_head = nn.Sequential(nn.Linear(hidden + skip_dim, hidden // 2), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden // 2, 1))

    def forward(self, Xc_seq, Xe_seq, A_seq):
        hc = self.comp_branch(Xc_seq, A_seq)
        he = self.exo_branch(Xe_seq, A_seq)
        hc = hc * self.comp_gate(hc)
        he = he * self.exo_gate(he)
        if self.use_gated_fusion:
            g = self.fuse_gate(torch.cat([hc, he], dim=-1))
            h = g * hc + (1.0 - g) * he
        else:
            h = 0.5 * hc + 0.5 * he
        if self.skip_proj is not None:
            skip_x = Xe_seq[:, -1, :, self.skip_idx_list]
            skip_h = self.skip_proj(skip_x)
            h_cat = torch.cat([h, skip_h], dim=-1)
        else:
            h_cat = h
        main_out = self.main_head(h_cat).squeeze(-1)
        event_logit = self.event_head(h_cat).squeeze(-1)
        calib_bias = self.calib_head(h_cat).squeeze(-1)
        pred = main_out + 0.15 * calib_bias
        return (pred, event_logit)

def build_model(hidden, dropout, ablation_variant='Full'):
    use_graph = True
    use_gated_fusion = True
    use_skip = True
    if ablation_variant == 'w/o Graph Convolution':
        use_graph = False
    if ablation_variant == 'w/o Gated Fusion':
        use_gated_fusion = False
    if ablation_variant == 'w/o Skip Connection':
        use_skip = False
    return DualBranchCEGNet(n_nodes=len(city_order), comp_dim=C_dim, exo_dim=F_dim, hidden=hidden, hist=history_len, skip_idx_list=skip_feature_idx, dropout=dropout, use_graph=use_graph, use_gated_fusion=use_gated_fusion, use_skip=use_skip).to(device)

class MultiObjectiveDualLoss(nn.Module):

    def __init__(self, peak_thr, event_focus_thr=100.0, w_mse=0.4, w_huber=0.25, w_smape=0.15, w_event=0.2, peak_weight=2.5, delta=1.0, use_event_loss=True):
        super().__init__()
        self.peak_thr = float(peak_thr)
        self.event_focus_thr = float(event_focus_thr)
        self.w_mse = float(w_mse)
        self.w_huber = float(w_huber)
        self.w_smape = float(w_smape)
        self.w_event = float(w_event)
        self.peak_weight = float(peak_weight)
        self.delta = float(delta)
        self.use_event_loss = use_event_loss

    def forward(self, pred, event_logit, target):
        peak_w = torch.ones_like(target)
        peak_w[target >= self.peak_thr] = self.peak_weight
        mse = (peak_w * (pred - target) ** 2).mean()
        err = pred - target
        abs_err = torch.abs(err)
        delta_tensor = torch.tensor(self.delta, device=target.device, dtype=target.dtype)
        quad = torch.minimum(abs_err, delta_tensor)
        lin = abs_err - quad
        huber = (peak_w * (0.5 * quad ** 2 + self.delta * lin)).mean()
        smape = torch.mean(2.0 * torch.abs(pred - target) / (torch.abs(pred) + torch.abs(target) + 0.001))
        if self.use_event_loss:
            event_target = (target >= self.event_focus_thr).float()
            event_loss = F.binary_cross_entropy_with_logits(event_logit, event_target)
        else:
            event_loss = torch.tensor(0.0, device=target.device, dtype=target.dtype)
        total = self.w_mse * mse + self.w_huber * huber + self.w_smape * smape + self.w_event * event_loss
        return total
