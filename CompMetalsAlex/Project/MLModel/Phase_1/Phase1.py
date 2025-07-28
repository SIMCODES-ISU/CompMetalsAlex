import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd

class Phase1Net(nn.Module):
    def __init__(self, hidden_dim=32):
        super(Phase1Net, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        return self.model(x)

class Phase1Dataset(Dataset):
    def __init__(self, df, transform=True):
        self.Sij = df["Sij"].values
        self.Rij = df["Rij"].values
        self.y = df["Damping value"].values
        self.transform = transform

    def __len__(self):
        return len(self.Sij)

    def __getitem__(self, idx):
        S = self.Sij[idx]
        R = self.Rij[idx]

        # Apply transformations to raw values of Sij and Rij for each system.
        # Transformations are inspired by the current EFP correction formula.
        # This is mainly to speed up training/convergence.
        if self.transform:
            x1 = -1.0 / np.log(S)
            x2 = (S*S) / R
        else:
            x1 = S
            x2 = R
        
        X = torch.tensor([x1, x2], dtype=torch.float64)
        y = torch.tensor(self.y[idx], dtype=torch.float64)

        return X, y

def calculate_binned_RMSE(model, dataset, destination):
    model.eval()
    true_vals = []
    pred_vals = []

    with torch.no_grad():
        for X, y in DataLoader(dataset, batch_size=1024):
            pred = model(X).squeeze()
            true_vals.extend(y.numpy())
            pred_vals.extend(pred.numpy())

    true_vals = np.array(true_vals)
    pred_vals = np.array(pred_vals)

    bins = [
        (1e-1, 1e-3),
        (1e-3, 1e-4),
        (1e-4, 1e-5),
        (1e-5, 1e-6),
        (1e-6, 1e-17)  # Adjust this to your minimum nonzero value
    ]

    bin_log = []

    for low, high in bins:
        mask = (np.abs(true_vals) <= low) & (np.abs(true_vals) > high)
        count = np.sum(mask)

        if count > 0:
            rmse = np.sqrt(np.mean((pred_vals[mask] - true_vals[mask])**2))
        else:
            rmse = np.nan

        bin_log.append({
            "Magnitude Bin": f"{low:.0e} to {high:.0e}",
            "Count": count,
            "RMSE": rmse
        })

    bin_df = pd.DataFrame(bin_log)
    bin_df.to_csv(destination, index=False)



if __name__ == "__main__":
    loss_log = []
    model = Phase1Net(hidden_dim=32).double()
    df = pd.read_csv("/mnt/c/c++_tests/games_work/ml_work/synth_data/Damped70750s90kr160kt.csv")
    dataset = Phase1Dataset(df)
    dataloader = DataLoader(dataset, batch_size=256, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr = 1e-4)
    loss_fn = nn.MSELoss()

    num_epochs = 200

    for epoch in range(num_epochs):
        total_loss = 0.0
        model.train()

        for i, (X, y) in enumerate(dataloader):
            # Reset gradients
            optimizer.zero_grad()

            #Get correction value for one Sij/Rij pair
            corrections = model(X)             

            # Compute loss
            loss = loss_fn(corrections.squeeze(), y)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * X.size(0)

        avg_loss = total_loss / len(dataset)
        loss_log.append({"epoch": epoch + 1, "total_loss": total_loss, "avg_loss": avg_loss})
        print(f"Epoch {epoch+1}: Total MSE Loss = {total_loss:.6f}\nAverage MSE Loss: {avg_loss:.10f}")

    loss_df = pd.DataFrame(loss_log)
    loss_df.to_csv("phase1_loss_log_200E_1-4LR_MSE_5.csv", index=False)
    torch.save(model.state_dict(), "phase1_damping_net.pth")
    torch.save(model, "phase1_damping_net_full.pt")
    calculate_binned_RMSE(model, dataset, "/mnt/c/c++_tests/games_work/ml_work/MLModel/Phase_1/Trial_5/binned_RMSE5.csv")