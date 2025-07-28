import os
import pandas as pd
import numpy as np
import scipy
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib
matplotlib.use('agg')



def graph_pairs(pair_data, plotname):

    # --- Load pair CSV ---
    pair_df = pd.read_csv(pair_data)
        
    # --- Plotting ---
    #plt.figure(figsize=(8,6))
    plt.figure(figsize=(16, 12)) 
    plt.scatter(pair_df['Sij'], pair_df['Rij'], alpha=0.1)
    plt.xlabel('Sij', fontsize=18)
    plt.ylabel('Rij', fontsize=18)
    plt.title('Sij vs Rij\n\n70750 synthetic points based on 90k real samples', fontsize=18)
    plt.xticks(np.arange(0, 0.312, 0.013), fontsize=14) #ticks every 0.013 units up to 0.312

# Add more ticks to the y-axis
    plt.yticks(np.arange(0, 25, 1), fontsize=14) # Ticks every 1 up to 24
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(plotname)



#Extract current total coulomb prediction
def total_electro_extract(output_directory, input_directory):
    output_file = output_directory + "Total_Coulomb.csv"
    with open(output_file, 'w') as out_file:
        out_file.write("File,Total Coulomb\n")
        for file_name in os.listdir(input_directory):
            if os.path.isfile(os.path.join(input_directory, file_name)):
                with open(input_directory + file_name, 'r') as in_file:
                    if file_name.endswith(".log"):
                        if file_name.startswith("NBC1") or file_name.startswith("HBC1"):
                            out_file.write(file_name[0:file_name.find("-C")] + ",")
                        else:
                            out_file.write(file_name[0:file_name.find("-unCP")] + ",")
                        for line in in_file:
                            line = line.strip()
                            if line.startswith("ELECTROSTATIC ENERGY"):
                                separate = line.split()
                                out_file.write(str(separate[3]) + "\n")


def calculate_errors_total_c(file1, file2, destination):
    df1 = pd.read_csv(file1)
    df2 = pd.read_csv(file2)
    merged_df = pd.merge(df1, df2, on='File', how='inner')
    merged_df["|SAPT - Total Undamped Coulomb|"] = abs(merged_df["SAPT Coulomb"] - merged_df["Total Coulomb"])
    merged_df.drop(['SAPT Coulomb', 'Undamped Coulomb', 'Total Coulomb'], axis=1, inplace=True)
    merged_df.to_csv(destination, index = False)

def calculate_errors_total_cby2(file1, file2, destination):
    df1 = pd.read_csv(file1)
    df2 = pd.read_csv(file2)
    df_merged = pd.merge(df1, df2, on='File', how='inner')
    df_merged = df_merged.dropna()
    df_merged["|SAPT - Total Coulomb EFP (damped*2)|"] = abs(df_merged["SAPT Coulomb_x"] - df_merged["Total Coulomb (damped*2)"])
    df_merged.drop(['SAPT Coulomb_x','SAPT Coulomb_y', 'Overlap Pen. Energy * 2', 'Undamped Coulomb_x', 'Undamped Coulomb_y', 'Total Coulomb (damped*2)'], axis=1, inplace=True)
    df_merged.to_csv(destination, index = False)

def calculate_errors_undamped(file, destination):
    df = pd.read_csv(file)
    df["|SAPT - Total Undamped Coulomb|"] = abs(df["SAPT Coulomb"] - df["Undamped Coulomb"])
    df.drop(['SAPT Coulomb', 'Undamped Coulomb'], axis=1, inplace=True)
    df.to_csv(destination, index = False)

def calculate_MAE(input_file):
    df = pd.read_csv(input_file)
    total_entries = len(df)
    total_error = df["|SAPT - Total Coulomb EFP (damped*2)|"].sum()
    MAE = total_error/total_entries
    print(f"Number of files: {total_entries}")
    print(f"The MAE is: {MAE}")

def calculate_RMSE(prediction_file):
    df1 = pd.read_csv(prediction_file)
    merged_df = df1
    total_entries = len(df1)
    merged_df["Difference"] = merged_df["SAPT Coulomb"] - (merged_df["Undamped Coulomb"] + merged_df["Prediction: OPE(56,000(OPE + 0.0302)^3 + 1.20)"])
    merged_df["Abs Difference"] = abs(merged_df["Difference"])
    merged_df["Squared Difference"] = merged_df["Difference"]**2
    total_squared = merged_df["Squared Difference"].sum()
    total_abs = merged_df["Abs Difference"].sum()
    MAE = total_abs/total_entries
    MSE = total_squared/total_entries
    RMSE = math.sqrt(MSE)
    print(f"Number of files: {total_entries}")
    print(f"The MAE is: {MAE}")
    print(f"The MSE is: {MSE}")
    print(f"The RMSE is: {RMSE}")


def graph_errors(error_csv_path, pair_data_dir, plotname):

    # --- Load error CSV ---
    error_df = pd.read_csv(error_csv_path)

    # --- Count number of pairs for each file ---
    pair_counts = []

    for filename in error_df['File']:
        file_path = os.path.join(pair_data_dir, f"{filename}.csv")
        
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            n_pairs = len(df)
            pair_counts.append(n_pairs)
        else:
            print(f"Warning: {filename}.csv not found")
            pair_counts.append(None)

    # Add pair count column to DataFrame
    error_df['n_pairs'] = pair_counts
    
    # --- Drop rows with missing pair counts (if any) ---
    error_df = error_df.dropna(subset=['n_pairs'])

    # --- Plotting ---
    plt.figure(figsize=(8,6))
    plt.scatter(error_df['n_pairs'], error_df['|SAPT - Total Undamped Coulomb|'], alpha=0.5)
    plt.xlabel('Number of Sij/Rij pairs')
    plt.ylabel('|SAPT - EFP(Damped)| (hartree)')
    plt.title('EFP Undamped Coulomb Error vs Number of Pairs (450 samples)')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(plotname)

def min_max_sij_rij(input_directory):
    min_sij = math.inf
    min_sij_syst = ""
    max_sij = -math.inf
    max_sij_syst = ""
    min_rij = math.inf
    min_rij_syst = ""
    max_rij = -math.inf
    max_rij_syst = ""
    for file_name in os.listdir(input_directory):
        if os.path.isfile(os.path.join(input_directory, file_name)):
            with open(input_directory + file_name, 'r') as in_file:              
                for line in in_file:
                    line = line.strip()
                    if line.startswith("Sij"):
                        continue
                    else:
                        fields = line.split(",")
                        fields = list(map(float, fields))
                        if fields[0] < min_sij:
                            min_sij = fields[0]
                            min_sij_syst = file_name
                        if fields[0] > max_sij:
                            max_sij = fields[0]
                            max_sij_syst = file_name
                        if fields[1] < min_rij:
                            min_rij = fields[1]
                            min_rij_syst = file_name
                        if fields[1] > max_rij:
                            max_rij = fields[1]
                            max_rij_syst = file_name
    print(f"Minimum Sij: {min_sij} in system {min_sij_syst}")
    print(f"Maximum Sij: {max_sij} in system {max_sij_syst}")
    print(f"Minimum Rij: {min_rij} in system {min_rij_syst}")
    print(f"Maximum Rij: {max_rij} in system {max_rij_syst}")  


def plot_sij_rij_vs_prediction(filename, plotname):

    sij_all = []
    rij_all = []
    pred_all = []

    df = pd.read_csv(filename)
    sij_raw = df['Sij'].values
    rij_raw = df['Rij'].values
    # Predictions
    preds = df['damping_value'].values

    n_pairs = len(df)

    # Plot the results
    fig = plt.figure(figsize=(20, 14))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(sij_raw, rij_raw, preds, alpha=0.2, s=10)
    ax.set_xlabel("Raw Sij")
    ax.set_ylabel("Raw Rij")
    ax.set_zlabel("Predicted Correction (existing function)")
    ax.set_title(f"Damping Values vs Raw Sij and Rij ({len(sij_raw)} points)")
    plt.savefig(plotname)



if __name__ == "__main__":
    #graph_errors("/mnt/c/c++_tests/games_work/ml_work/clean_data/Ground_truth_files/Baselines/Errors_undamped_450.csv","/mnt/c/c++_tests/games_work/ml_work/clean_data/All_data/", "/mnt/c/c++_tests/games_work/ml_work/MLModel/Neural net graphs/Baselines 450 samples/450sampAbsErrEFPvsSAPTUndamped.png")
    #total_electro_extract("/mnt/c/c++_tests/games_work/ml_work/clean_data/Ground_truth_files/", "/mnt/c/c++_tests/games_work/ml_work/Raw_data/All files/")
    #calculate_errors("/mnt/c/c++_tests/games_work/ml_work/clean_data/Ground_truth_files/meta.csv", "/mnt/c/c++_tests/games_work/ml_work/clean_data/Ground_truth_files/Baselines/Total_Coulomb.csv", "/mnt/c/c++_tests/games_work/ml_work/clean_data/Ground_truth_files/Baselines/Errors_total.csv")
    #min_max_sij_rij("/mnt/c/c++_tests/games_work/ml_work/clean_data/All_data/")
    #calculate_errors_total_cby2("/mnt/c/c++_tests/games_work/ml_work/clean_data/Ground_truth_files/meta.csv", "/mnt/c/c++_tests/games_work/ml_work/clean_data/Ground_truth_files/total_coulomb_dampby2_450.csv", "/mnt/c/c++_tests/games_work/ml_work/clean_data/Ground_truth_files/Baselines/Errors_total_dampby2_450.csv")
    #calculate_MAE("/mnt/c/c++_tests/games_work/ml_work/clean_data/Ground_truth_files/Baselines/Errors_total_dampby2.csv")
    calculate_RMSE("/mnt/c/c++_tests/games_work/ml_work/MLModel/SymbolicReg/Trial_3/testdatax_predictions1.csv")
    #graph_pairs("/mnt/c/c++_tests/games_work/ml_work/synth_data/70kplus750on90k_filtered01.csv", "/mnt/c/c++_tests/games_work/ml_work/Graphs/70750syn_on90k02.png")
    #calculate_errors_total_c("/mnt/c/c++_tests/games_work/ml_work/clean_data/Ground_truth_files/meta.csv", "/mnt/c/c++_tests/games_work/ml_work/clean_data/Ground_truth_files/meta.csv", "/mnt/c/c++_tests/games_work/ml_work/clean_data/Ground_truth_files/Baselines/Errors_undamped_450.csv")
    #plot_sij_rij_vs_prediction("/mnt/c/c++_tests/games_work/ml_work/MLModel/SymbolicReg/data/test1/all_damped_data.csv", "/mnt/c/c++_tests/games_work/ml_work/MLModel/Neural net graphs/Baselines 329 samples/SijRijDamping.png")
    #calculate_errors_undamped("/mnt/c/c++_tests/games_work/ml_work/clean_data/Ground_truth_files/meta.csv", "/mnt/c/c++_tests/games_work/ml_work/clean_data/Ground_truth_files/Baselines/Errors_undamped_450.csv")

    # df = pd.read_csv("/mnt/c/c++_tests/games_work/ml_work/MLModel/SymbolicReg/Trial_3/testdatax2.csv")
    # df["Prediction: OPE(56,000(OPE + 0.0302)^3 + 1.20)"] = df["Overlap Pen. Energy"]*(56000*((df["Overlap Pen. Energy"] + 0.0302)**3) + 1.2)
    # df.to_csv("/mnt/c/c++_tests/games_work/ml_work/MLModel/SymbolicReg/Trial_3/testdatax_predictions1.csv", index=False)

