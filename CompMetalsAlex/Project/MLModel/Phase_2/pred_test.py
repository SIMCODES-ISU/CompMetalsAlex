import sys
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import numpy as np
import pandas as pd
phase1_dir = os.path.abspath('/mnt/c/c++_tests/games_work/ml_work/MLModel/Phase_1')
sys.path.append(phase1_dir)
from Phase1 import Phase1Net

def print_predictions(destination):
    df = pd.read_csv("/mnt/c/c++_tests/games_work/ml_work/MLModel/SymbolicReg/data/test1/all_damped_data.csv")

    x1 = -1.0 / np.log(df["Sij"].values)
    x2 = (df["Sij"].values ** 2) / df["Rij"].values
    X = torch.tensor(np.column_stack([x1, x2]), dtype=torch.float64)

    model = Phase1Net(hidden_dim=32).double() 
    model.load_state_dict(torch.load("phase2_damping_net.pth"))
    model.eval()

    with torch.no_grad():
        damping_preds = model(X).squeeze()

    df["NN Prediction"] = damping_preds.detach().numpy()
    df["|Base - Pred|"] = abs(df["damping_value"] - df["NN Prediction"])
    df["MSE: (Base-Pred)**2)"] = (df["damping_value"] - df["NN Prediction"])**2
    df.to_csv(destination, index=False)

def NN_damp(input_directory, outfile):

    model = Phase1Net(hidden_dim=32).double() 
    model.load_state_dict(torch.load("/mnt/c/c++_tests/games_work/ml_work/MLModel/Phase_2/Trial_12/phase2_damping_net.pth"))
    model.eval()

    with open(outfile, 'w') as out_file:
        out_file.write("File,DampNN\n")
        for filename in os.listdir(input_directory):
            out_file.write(filename[0:filename.find(".csv")] + ",")
            # with open(input_directory + filename, 'r') as in_file:
            in_file = pd.read_csv(input_directory + filename)
            x1 = -1.0 / np.log(in_file["Sij"].values)
            x2 = (in_file["Sij"].values ** 2) / in_file["Rij"].values
            X = torch.tensor(np.column_stack([x1, x2]), dtype=torch.float64)
            with torch.no_grad():
                damping_preds = model(X).squeeze()
                NN_damp_value = damping_preds.sum().item()
            out_file.write(str(NN_damp_value) + "\n")


if __name__ == "__main__":
    #print_predictions("/mnt/c/c++_tests/games_work/ml_work/MLModel/Phase_1/Trial_1/predictions_file.csv")
    #NN_damp("/mnt/c/c++_tests/games_work/ml_work/clean_data/All_data/", "/mnt/c/c++_tests/games_work/ml_work/MLModel/Phase_2/Trial_12/NN_per_system12.csv")

    df = pd.read_csv("/mnt/c/c++_tests/games_work/ml_work/MLModel/Phase_2/Trial_12/SAPT_Undp_NNcorrections12.csv")
    df["Total Coulomb"] = df["Undamped Coulomb"] + df["DampNN"]
    df.to_csv("/mnt/c/c++_tests/games_work/ml_work/MLModel/Phase_2/Trial_12/SAPT_Undp_NNcorrections12.csv", index=False)

