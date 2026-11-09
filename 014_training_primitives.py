def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0.0
    for Xcb, Xeb, Yb, Ab in loader:
        Xcb = Xcb.to(device)
        Xeb = Xeb.to(device)
        Yb = Yb.to(device)
        Ab = Ab.to(device)
        optimizer.zero_grad()
        pred, event_logit = model(Xcb, Xeb, Ab)
        loss = criterion(pred, event_logit, Yb)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        total_loss += loss.item() * Xcb.size(0)
    return total_loss / len(loader.dataset)

def evaluate(model, loader, criterion):
    model.eval()
    total_loss = 0.0
    preds, trues = ([], [])
    with torch.no_grad():
        for Xcb, Xeb, Yb, Ab in loader:
            Xcb = Xcb.to(device)
            Xeb = Xeb.to(device)
            Yb = Yb.to(device)
            Ab = Ab.to(device)
            pred, event_logit = model(Xcb, Xeb, Ab)
            loss = criterion(pred, event_logit, Yb)
            total_loss += loss.item() * Xcb.size(0)
            preds.append(pred.cpu().numpy())
            trues.append(Yb.cpu().numpy())
    preds = np.vstack(preds)
    trues = np.vstack(trues)
    return (total_loss / len(loader.dataset), preds, trues)

def predict_only(model, loader):
    model.eval()
    preds = []
    with torch.no_grad():
        for Xcb, Xeb, Ab in loader:
            Xcb = Xcb.to(device)
            Xeb = Xeb.to(device)
            Ab = Ab.to(device)
            pred, _ = model(Xcb, Xeb, Ab)
            preds.append(pred.cpu().numpy())
    return np.vstack(preds)
