class DualBranchAQIDataset(Dataset):

    def __init__(self, Xc, Xe, Y=None, A=None):
        self.Xc = torch.tensor(Xc, dtype=torch.float32)
        self.Xe = torch.tensor(Xe, dtype=torch.float32)
        self.Y = torch.tensor(Y, dtype=torch.float32) if Y is not None else None
        self.A = torch.tensor(A, dtype=torch.float32) if A is not None else None

    def __len__(self):
        return len(self.Xc)

    def __getitem__(self, idx):
        if self.Y is None:
            return (self.Xc[idx], self.Xe[idx], self.A[idx])
        return (self.Xc[idx], self.Xe[idx], self.Y[idx], self.A[idx])
