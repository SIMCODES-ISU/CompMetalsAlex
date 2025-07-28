import os
import json
import torch
from tqdm import tqdm
import torch.nn as nn
from torch.utils.data import Dataset
import pandas as pd
import random
# import matplotlib.pyplot as plt
# import matplotlib
# matplotlib.use('agg')

class EFPCorrectionDataset(Dataset):
    def __init__(self, csv_dir, metadata_path):
        """
        Args:
            csv_dir (str): Path to directory with per-system CSV files.
            metadata_path (str): Path to meta.json file.
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
        Sij = torch.tensor(df['Sij'].values, dtype=torch.float32)
        Rij = torch.tensor(df['Rij'].values, dtype=torch.float32)

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

        # system_id = self.system_ids[idx]
        # csv_path = os.path.join(self.csv_dir, f"{system_id}.csv")

        # # Load Sij and Rij pairs
        # df = pd.read_csv(csv_path)

        # # shape: [n_pairs, 2].
        # #There is one matrix per system with "n_pairs" rows and 2 columns.
        # x_pairs = torch.tensor(df.to_numpy(), dtype=torch.float32)  

        # # Get per-system scalars from metadata
        # E_undamped = torch.tensor(self.metadata[system_id]["Undamped Coulomb"], dtype=torch.float32)
        # E_sapt = torch.tensor(self.metadata[system_id]["SAPT Coulomb"], dtype=torch.float32)

        # return x_pairs, E_undamped, E_sapt, system_id


class EFPNet(nn.Module):
    def __init__(self, hidden_dim=8):
        super(EFPNet, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        return self.model(x)  # x: [n_pairs, 2] → output: [n_pairs, 1]

if __name__ == "__main__":

    # dataset = EFPCorrectionDataset(csv_dir="/mnt/c/c++_tests/games_work/ML_work/clean_data/All_data/", metadata_path="/mnt/c/c++_tests/games_work/ML_work/clean_data/Ground_truth_files/meta.json")
    # idx = random.randint(0, len(dataset) - 1)
    # x, E_undamped, E_sapt, name = dataset[idx]
    # print(f"System ID: {name}")
    # print(f"Number of pairs: {x.shape[0]}")
    # print(f"Transformed Sij and Rij pairs (first 5 rows):\n{x[:5]}")
    # print(f"Undamped energy: {E_undamped.item()}")
    # print(f"SAPT ground truth: {E_sapt.item()}")

    bad_Sij = torch.tensor([-0.05, 0.01, 0.000000000001, 0.03])  # Includes 0
    bad_Rij = torch.tensor([5.0, 0.0, 7.0, 8.0])
    x_pairs = torch.stack([bad_Sij, bad_Rij], dim=1)
    print(x_pairs)

    if (bad_Sij <= 0).any():
        raise ValueError("Test: Sij contains zero or negative values")
    if (bad_Rij <= 0).any():
        raise ValueError("Test: Rij contains zero or negative values")
    
    
    
    
    
    
    #! model = EFPNet(hidden_dim=8)
    # dataset = EFPCorrectionDataset(csv_dir="/mnt/c/c++_tests/games_work/ML_work/clean_data/All_data/", metadata_path="/mnt/c/c++_tests/games_work/ML_work/clean_data/Ground_truth_files/meta.json")
    # optimizer = torch.optim.Adam(model.parameters(), lr = 1e-3)
    # loss_fn = nn.MSELoss()

    # num_epochs = 100

    # for epoch in range(num_epochs):
    #     total_loss = 0.0

    #     for x_pairs, E_undamped, E_sapt, system_id in tqdm(dataset, desc=f"Epoch {epoch+1}"):
    #         # Reset gradients
    #         optimizer.zero_grad()

    #         # Forward pass: get correction values for all pairs
    #         correction_terms = model(x_pairs)             # shape [n_pairs, 1]
    #         total_correction = correction_terms.sum()     # scalar

    #         # Predicted total energy
    #         E_predicted = E_undamped + total_correction

    #         # Compute loss
    #         loss = loss_fn(E_predicted, E_sapt)

    #         # Backward pass and optimize
    #         loss.backward()
    #         optimizer.step()

    #         total_loss += loss.item()

    #     print(f"Epoch {epoch+1}: Total MSE Loss = {total_loss:.6f}")



    # model.eval()
    # errors = []

    # with torch.no_grad():
    #     for x_pairs, E_undamped, E_sapt, system_id in dataset:
    #         correction_terms = model(x_pairs)
    #         total_correction = correction_terms.sum()
    #         E_predicted = E_undamped + total_correction

    #         error = abs(E_predicted.item() - E_sapt.item())
    #         errors.append((system_id, error))

    # # Sort and print worst offenders
    # errors.sort(key=lambda x: x[1], reverse=True)

    # print("\nTop 10 largest prediction errors:")
    # for system_id, err in errors[:10]:
    #!     print(f"{system_id}: Error = {err:.4f} hartree")


    # sample_indices = [0, 1, 2, 3, 4]  # or any other subset

    # model.eval()  # put model in evaluation mode

    # with torch.no_grad():  # turn off gradient tracking
    #     for idx in sample_indices:
    #         x_pairs, E_undamped, E_sapt, system_id = dataset[idx]

    #         correction_terms = model(x_pairs)  # shape [n_pairs, 1]
    #         total_correction = correction_terms.sum()

    #         E_predicted = E_undamped + total_correction

    #         print(f"System: {system_id}")
    #         print(f"  Undamped:  {E_undamped.item():.8f} hartree")
    #         print(f"  Predicted: {E_predicted.item():.8f} hartree")
    #         print(f"  SAPT GT:   {E_sapt.item():.8f} hartree")
    #         print(f"  Error:     {(E_predicted - E_sapt).item():.8f} hartree")
    #         print("-" * 50)    
    





















#     dataset = EFPCorrectionDataset(csv_dir="/mnt/c/c++_tests/games_work/ML_work/clean_data/All_data/", metadata_path="/mnt/c/c++_tests/games_work/ML_work/clean_data/Ground_truth_files/meta.json")
#     print(f"Number of systems in dataset: {len(dataset)}")

#     x_pairs, E_undamped, E_sapt, system_id = dataset[1]

#     print(f"System ID: {system_id}")
#     print(f"Number of pairs: {x_pairs.shape[0]}")
#     print(f"Sij and Rij pairs (first 5 rows):\n{x_pairs[:5]}")
#     print(f"Undamped energy: {E_undamped.item()}")
#     print(f"SAPT ground truth: {E_sapt.item()}")

#     idx = random.randint(0, len(dataset) - 1)
#     x, E_undamped, E_sapt, name = dataset[idx]
#     print(f"System ID: {name}")
#     print(f"Number of pairs: {x.shape[0]}")
#     print(f"Sij and Rij pairs (first 5 rows):\n{x[:5]}")
#     print(f"Undamped energy: {E_undamped.item()}")
#     print(f"SAPT ground truth: {E_sapt.item()}")

#     sij = x[:, 0].numpy()
#     rij = x[:, 1].numpy()
#     plt.scatter(rij, sij, alpha=0.6)
#     plt.xlabel("R_ij (distance)")
#     plt.ylabel("S_ij (overlap)")
#     plt.title(f"{name} - Pairwise Inputs")
#     plt.show()