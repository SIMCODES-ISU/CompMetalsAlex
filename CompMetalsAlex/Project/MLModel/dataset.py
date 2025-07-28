import os
import json
import torch
from tqdm import tqdm
import torch.nn as nn
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import math
import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib
matplotlib.use('agg')

class EFPCorrectionDataset(Dataset):
    def __init__(self, csv_dir, metadata_path):
        """
        Args:
            csv_dir (str): Path to directory with per-system CSV files.
            metadata_path (str): Path to meta.json file (contains all ground truth values).
        """
        self.csv_dir = csv_dir

        # Load JSON metadata (all ground truth values are in meta.json)
        with open(metadata_path, "r") as f:
            self.metadata = json.load(f)

        # Only include systems for which a matching CSV exists
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

        # Load the Sij and Rij columns
        df = pd.read_csv(csv_path)
        Sij = torch.tensor(df['Sij'].to_numpy(), dtype=torch.float32)
        Rij = torch.tensor(df['Rij'].to_numpy(), dtype=torch.float32)

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
        E_undamped = torch.tensor(self.metadata[system_id]["Undamped Coulomb"], dtype=torch.float32)
        E_sapt = torch.tensor(self.metadata[system_id]["SAPT Coulomb"], dtype=torch.float32)

        return x_pairs, E_undamped, E_sapt, system_id


class EFPNet(nn.Module):
    def __init__(self, hidden_dim=8):
        super(EFPNet, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        return self.model(x)  # x: [n_pairs, 2] -> output: [n_pairs, 1]


def plot_error_vs_pair_count(model, dataset, plotname, lr, epochs, hidden_dim, total_loss, samples):
    model.eval()
    num_pairs_list = []
    error_list = []
    system_ids = []

    with torch.no_grad():
        for x_pairs, E_undamped, E_sapt, system_id in dataset:
            # Predict per-pair corrections
            corrections = model(x_pairs)
            total_correction = corrections.sum()

            # Predicted total electrostatic energy
            E_pred = E_undamped + total_correction
            error = torch.abs(E_pred - E_sapt).item()

            # Log data
            num_pairs_list.append(x_pairs.shape[0])
            error_list.append(error)
            system_ids.append(system_id)

    # Convert to numpy for easy handling
    num_pairs_list = np.array(num_pairs_list)
    error_list = np.array(error_list)

    # Plot
    plt.figure(figsize=(8, 6))
    plt.scatter(num_pairs_list, error_list, alpha=0.7)
    plt.xlabel("Number of Pairs")
    plt.ylabel("Absolute Error (hartree)")
    plt.title("Prediction Error vs Number of Sij/Rij Pairs")
    plt.grid(True)

    textstr = f"LR = {lr}\nEpochs = {epochs}\nHiddenDim = {hidden_dim}\nTotalMSE = {total_loss:.4f}\nSamples = {samples}\nAvg_MSE = {math.sqrt(total_loss/int(samples)):.4f}"

    plt.gca().text(
    0.15, 0.95, textstr,
    transform = plt.gca().transAxes,  # Use axes coordinates (0 to 1)
    fontsize = 8,
    verticalalignment = 'top',
    horizontalalignment = 'left',
    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", edgecolor="black")
    )

    plt.tight_layout()
    plt.savefig(plotname)

def plot_sij_rij_vs_prediction(model, data_dir, plotname, num_points=1000):
    model.eval()

    sij_all = []
    rij_all = []
    pred_all = []

    # Collect list of filenames
    filenames = os.listdir(data_dir)
    random.shuffle(filenames)

    with torch.no_grad():
        for fname in filenames:
            if len(sij_all) >= num_points:
                break

            file_path = os.path.join(data_dir, fname)
            df = pd.read_csv(file_path)

            sij_raw = df['Sij'].values
            rij_raw = df['Rij'].values

            # Transform features just like in training
            # (-1 / ln(Sij), Sij^2 / Rij)

            sij_tensor = torch.tensor(sij_raw, dtype=torch.float32)
            rij_tensor = torch.tensor(rij_raw, dtype=torch.float32)

            # Error handling: skip bad values
            if (sij_tensor <= 0).any() or (rij_tensor == 0).any():
                continue

            x1 = -1.0 / torch.log(sij_tensor)
            x2 = (sij_tensor * sij_tensor) / rij_tensor
            x_pairs = torch.stack([x1, x2], dim=1)

            # Predict
            preds = model(x_pairs).squeeze().numpy()

            # Downsample if too many points
            n_pairs = len(sij_raw)
            if len(sij_all) + n_pairs > num_points:
                keep = num_points - len(sij_all)
                indices = np.random.choice(n_pairs, keep, replace=False)
            else:
                indices = np.arange(n_pairs)

            for idx in indices:
                sij_all.append(sij_raw[idx])
                rij_all.append(rij_raw[idx])
                pred_all.append(preds[idx])

    # Plot the results
    fig = plt.figure(figsize=(20, 14))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(sij_all, rij_all, pred_all, alpha=0.6, s=10)
    ax.set_xlabel("Raw Sij")
    ax.set_ylabel("Raw Rij")
    ax.set_zlabel("Predicted Correction")
    ax.set_title(f"Model Predictions vs Raw Sij and Rij ({len(sij_all)} points)")
    plt.savefig(plotname)


if __name__ == "__main__":

    model = EFPNet(hidden_dim=16)
    dataset = EFPCorrectionDataset(csv_dir="/mnt/c/c++_tests/games_work/ML_work/clean_data/All_data/", metadata_path="/mnt/c/c++_tests/games_work/ML_work/clean_data/Ground_truth_files/meta.json")
    optimizer = torch.optim.Adam(model.parameters(), lr = 1e-4)
    loss_fn = nn.MSELoss()

    num_epochs = 200

    for epoch in range(num_epochs):
        total_loss = 0.0

        for x_pairs, E_undamped, E_sapt, system_id in tqdm(dataset, desc=f"Epoch {epoch+1}"):
            # Reset gradients
            optimizer.zero_grad()

            # Forward pass: get correction values for all pairs
            correction_terms = model(x_pairs)             # shape [n_pairs, 1]
            total_correction = correction_terms.sum()     # scalar

            # Predicted total energy
            E_predicted = E_undamped + total_correction

            # Compute loss
            loss = loss_fn(E_predicted, E_sapt)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}: Total MSE Loss = {total_loss:.6f}")

    print(f"The average RMSE loss per system by the last epoch was: {math.sqrt(total_loss/450)}")

    model.eval()
    errors = []

    with torch.no_grad():
        for x_pairs, E_undamped, E_sapt, system_id in dataset:
            correction_terms = model(x_pairs)
            total_correction = correction_terms.sum()
            E_predicted = E_undamped + total_correction

            error = abs(E_predicted.item() - E_sapt.item())
            errors.append((system_id, error, total_correction))

    # Sort and print worst offenders
    errors.sort(key=lambda x: x[1], reverse=True)

    print("\nTop 10 largest prediction errors:")
    for system_id, err,total_corr in errors[:10]:
        print(f"{system_id}: Error = {err:.4f} hartree")
  

    path_graphs = "/mnt/c/c++_tests/games_work/ml_work/MLModel/Neural net graphs/"
    data_directory = "/mnt/c/c++_tests/games_work/ml_work/clean_data/All_data/"
    plot_error_vs_pair_count(model, dataset, path_graphs + "lr-4_16dim_200E_450samp_2.png", "0.0001", "200", "16", total_loss, "450")
    plot_sij_rij_vs_prediction(model, data_directory, path_graphs + "lr-4_16dim200E450samp_predictions_vs_sijrij_2.png", 60000)