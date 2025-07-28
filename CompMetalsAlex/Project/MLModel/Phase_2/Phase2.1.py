import os
import sys
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import random_split
import torch.nn.functional as F
import numpy as np
import pandas as pd
import math
phase1_dir = os.path.abspath('/mnt/c/c++_tests/games_work/ml_work/MLModel/Phase_1')
sys.path.append(phase1_dir)
from Phase1 import Phase1Net

class SystemLevelDataset(Dataset):
    def __init__(self, csv_dir, metadata_path):
        self.csv_dir = csv_dir  # list of per-system .csvs
        
        with open(metadata_path, "r") as f:
            self.metadata = json.load(f)

        # Only include systems for which a matching CSV exists
        # There's a few files in database with SAPT data but no pair data
        all_ids = self.metadata.keys()
        self.system_ids = sorted([
            sys_id for sys_id in all_ids
            if os.path.isfile(os.path.join(csv_dir, f"{sys_id}.csv"))
        ])

    def __len__(self):
        return len(self.system_ids)

    def __getitem__(self, idx):
        
        system_id = self.system_ids[idx]
        csv_path = os.path.join(self.csv_dir, system_id + ".csv")

        df = pd.read_csv(csv_path)
        Sij = torch.tensor(df['Sij'].to_numpy(), dtype=torch.float64)
        Rij = torch.tensor(df['Rij'].to_numpy(), dtype=torch.float64)

        #Raise errors if invalid values are detected
        if(Sij<=0).any():
            raise ValueError(f"Sij contains zero or negative values in system {system_id}")
        if(Rij<=0).any():
            raise ValueError(f"Rij contains zero or negative values in system {system_id}")

        # Apply transformations to raw values of Sij and Rij for each system.
        # Transformations are inspired by the current EFP correction formula.
        x1 = -1.0 / torch.log(Sij)           # -1 / ln(Sij)
        x2 = (Sij * Sij) / Rij                      # Sij^2 / Rij
        x_pairs = torch.stack([x1, x2], dim=1)     # Shape: [n_pairs, 2]

        # Load target values
        E_undamped = torch.tensor(self.metadata[system_id]["Undamped Coulomb"], dtype=torch.float64)
        E_sapt = torch.tensor(self.metadata[system_id]["SAPT Coulomb"], dtype=torch.float64)

        return x_pairs, E_undamped, E_sapt, system_id

if __name__ == "__main__":
    loss_log = []
    #Load previous frozen NN from phase 1
    model = Phase1Net(hidden_dim=32).double()
    model.load_state_dict(torch.load("/mnt/c/c++_tests/games_work/ml_work/MLModel/Phase_1/Trial_5/phase1_damping_net.pth"))

    #Freeze layers
    for name, param in model.named_parameters():
        if "model.0" in name:  # Freeze first 1 layer
            param.requires_grad = False

    dataset = SystemLevelDataset(csv_dir="/mnt/c/c++_tests/games_work/ML_work/clean_data/All_data/", metadata_path="/mnt/c/c++_tests/games_work/ML_work/clean_data/Ground_truth_files/meta.json")
    
    val_size = int(0.1 * len(dataset))
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1)
    
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)

    num_epochs = 200
    alpha = 1.0
    beta  = 1.0
    scale = 1.0

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0

        for x_pairs, E_undamped, E_sapt, system_id in train_loader:
            optimizer.zero_grad()

            # NN outputs for all pairs
            correction_terms = model(x_pairs)  # shape [n_pairs]
            total_correction = correction_terms.sum()      # scalar

            # Target residual
            residual = E_sapt - E_undamped

            # Scale (optional but recommended if residuals ~1e-10)
            total_corr_s = total_correction * scale
            residual_s   = residual * scale

            # Base MSE
            #mse = F.mse_loss(total_corr_s, residual_s.squeeze())

            # Directional penalty (under-damped: E_pred > E_sapt)
            E_pred_s = (E_undamped + total_correction) * scale
            err = E_pred_s - (E_sapt * scale)
            dir_penalty = torch.relu(err)  # scalar

            # Pair-level positivity penalty
            neg_penalty = torch.relu(correction_terms).pow(2).mean()

            # Total loss
            loss = loss_fn(total_correction, residual.squeeze()) + alpha * dir_penalty + beta * neg_penalty
            total_loss += loss.item()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
        # Validation loop
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x_pairs, E_undamped, E_sapt, system_id in val_loader:
                correction_terms = model(x_pairs) # shape [n_pairs]
                total_correction = correction_terms.sum()      # scalar
                # Target residual
                residual = E_sapt - E_undamped
                # Scale (optional but recommended if residuals ~1e-10)
                total_corr_s = total_correction * scale
                residual_s   = residual * scale
                # Base MSE
                #mse = F.mse_loss(total_corr_s, residual_s.squeeze())
                # Directional penalty (under-damped: E_pred > E_sapt)
                E_pred_s = (E_undamped + total_correction) * scale
                err = E_pred_s - (E_sapt * scale)
                dir_penalty = torch.relu(err)  # scalar
                # Pair-level positivity penalty
                neg_penalty = torch.relu(correction_terms).pow(2).mean()
                # Total loss
                loss = loss_fn(total_correction, residual.squeeze()) + alpha * dir_penalty + beta * neg_penalty
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        avg_train_loss = total_loss / len(train_loader)
        loss_log.append({
        "epoch": epoch + 1,
        "train_total_loss": total_loss,
        "train_avg_loss": avg_train_loss,
        "val_total_loss": val_loss,
        "val_avg_loss": avg_val_loss
        })
        print(f"Epoch {epoch+1}:")
        print(f"System-level training MSE Loss = {total_loss:.6f}")
        print(f"Average training MSE Loss = {avg_train_loss:.6f}")
        print(f"System-level val. MSE Loss = {val_loss:.6f}")
        print(f"Average val. MSE Loss = {avg_val_loss:.6f}")


    avg_mse = total_loss / len(train_loader)
    RMSE = math.sqrt(avg_mse)
    print(f"Final RMSE per system: {RMSE:.6f}")
    loss_df = pd.DataFrame(loss_log)
    loss_df.to_csv("phase2_loss_log_200E_1E-4LR_MSE_Clipped1_Fr1Layer_penalties13.csv", index=False)
    torch.save(model.state_dict(), "phase2_damping_net.pth")
    torch.save(model, "phase2_damping_net_full.pt")