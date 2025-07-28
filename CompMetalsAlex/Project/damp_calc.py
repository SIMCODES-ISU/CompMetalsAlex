import pandas as pd
import math
import os

def damping(Sij, Rij):
    return -1 * math.sqrt( (1 / (-2 * math.log(abs(Sij))) ) ) * ( (Sij*Sij)/Rij )

def damping_pd(row):
    return -1 * math.sqrt( (1 / (-2 * math.log(abs(row["Sij"]))) ) ) * ( (row["Sij"]*row["Sij"])/row["Rij"] )


def damp_by2_calc(input_directory, output_file):
    with open(output_file, 'w') as out_file:
        out_file.write("File,Overlap Pen. Energy\n")
        for filename in os.listdir(input_directory):
            if filename.startswith("NBC1") or filename.startswith("HBC1"):
                out_file.write(filename[0:filename.find("-C")] + ",")
            else:
                out_file.write(filename[0:filename.find("-unCP")] + ",")
            with open(input_directory + filename, 'r') as in_file:
                for line in in_file:
                    line = line.strip()
                    if line.startswith("OVERLAP PEN."):
                        fields = line.split()
                        out_file.write(str(found_func(float(fields[4]))) + "\n")


#Create file with damp correction * 2 + undamped
def total_coulomb_dampby2(dampby2_source, undamped_source, destination):
    df1 = pd.read_csv(dampby2_source)
    df2 = pd.read_csv(undamped_source)
    df_merged = pd.merge(df1, df2, on='File', how='left')
    #Drop rows missing entries of either kind
    df_merged = df_merged.dropna()
    #Create new column with damp*2 + undamped
    df_merged["Total Coulomb (damped*2)"] = df_merged["Undamped Coulomb"] + df_merged["Overlap Pen. Energy * 2"]
    #Remove other columns
    df_merged.drop(["Undamped Coulomb", "Overlap Pen. Energy * 2"], axis='columns')
    #Save File
    df_merged.to_csv(destination, index = False)


def extract_and_damp(input_directory, output_directory):
    all_rows = []
    for filename in os.listdir(input_directory):
        if filename.endswith(".csv"):
            df = pd.read_csv(os.path.join(input_directory, filename))
            for sij, rij in zip(df["Sij"], df["Rij"]):
                if sij > 0 and rij > 0:
                    value = damping(sij, rij)
                    all_rows.append({"Sij": sij, "Rij": rij, "damping_value": value})

    # Limit to 1000 for proof of concept
    pd.DataFrame(all_rows).to_csv(output_directory + "all_damped_data.csv", index=False)
    #pd.DataFrame(all_rows[:1000]).to_csv(output_directory + "small_test_1000.csv", index=False)

def damp_one_file(file):
    df = pd.read_csv(file)
    damp_value = 0
    for index, row in df.iterrows():
        damp_value += damping(row["Sij"], row["Rij"])
    print(f"Damping value: {damp_value}")

def damp_database(file, dest):
    df = pd.read_csv(file)
    df["Damping Value Pred"] = df["Overlap Pen. Energy"].apply(found_func)
    df["Total Coulomb Pred"] = df["Undamped Coulomb"] + df["Damping Value Pred"]
    #df.drop('damping_value', axis=1, inplace=True) 
    df.to_csv(dest, index=False)

def found_func(overlap):
    return overlap - ( 4* (1e-5) / (overlap - 0.000001) )

if __name__ == "__main__":
    #extract_and_damp("/mnt/c/c++_tests/games_work/ml_work/clean_data/All_data/", "/mnt/c/c++_tests/games_work/ml_work/MLModel/SymbolicReg/data/test1/")
    #damp_by2_calc("/mnt/c/c++_tests/games_work/ml_work/Raw_data/All files/", "/mnt/c/c++_tests/games_work/ml_work/MLModel/SymbolicReg/data/overlap_pen_modified.csv")
    #total_coulomb_dampby2("/mnt/c/c++_tests/games_work/ml_work/clean_data/Ground_truth_files/Damped_by_2_450.csv", "/mnt/c/c++_tests/games_work/ml_work/clean_data/Ground_truth_files/meta.csv", "/mnt/c/c++_tests/games_work/ml_work/clean_data/Ground_truth_files/total_coulomb_dampby2_450.csv")
    #damp_one_file("/mnt/c/c++_tests/games_work/ml_work/clean_data/HBC1/HBC1-FaOOFaOO-4.8.csv")
    damp_database("/mnt/c/c++_tests/games_work/ml_work/MLModel/SymbolicReg/data/testset_real1/all_test_files.csv","/mnt/c/c++_tests/games_work/ml_work/MLModel/SymbolicReg/data/testset_real1/all_test_files_0_rounded2.csv")

    # df = pd.read_csv("/mnt/c/c++_tests/games_work/ml_work/clean_data/Ground_truth_files/meta.csv")
    # df["SAPT - Undamped"] = df["SAPT Coulomb"] - df["Undamped Coulomb"]
    # df.to_csv("/mnt/c/c++_tests/games_work/ml_work/clean_data/Ground_truth_files/signed_errors.csv", index = False)