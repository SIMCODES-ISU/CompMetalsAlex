import os
import pandas as pd
import json
import csv

"""
This file contains a series of utility functions to extract SAPT and undamped
data from incoming databases, and putting them neatly into .csv files. 
There's also a function that builds a master JSON file containing all SAPT
ground truth points and undamped electrostatic energy data. 
The JSON file allows for easy ground truth comparisons during neural
network training runs.
"""



#Calculate and extract undamped electrostatic energy for current files
#This function is somewhat "deprecated"
def undamped():
    output_file = output_directory + "undamped.csv"
    with open(output_file, 'w') as out_file:
        out_file.write("File,Undamped Coulomb\n")

        for dirpath, dirnames, filenames in os.walk(input_directory):
            # dirpath is the current directory being walked
            # dirnames is a list of subdirectories in dirpath
            # filenames is a list of files in dirpath
            for file_name in filenames:
                if os.path.isfile(os.path.join(dirpath, file_name)):
                    with open(os.path.join(dirpath, file_name), 'r') as in_file:
                        if file_name.endswith(".log"):
                            out_file.write(file_name[0:file_name.find("-unCP")] + ",")
                            undamped_coul = 0
                            for line in in_file:
                                line = line.strip()
                                if line.startswith("CHARGE-CHARGE"):
                                    undamped_coul += float(line[25:])
                                elif line.startswith("CHARGE-DIPOLE"):
                                    undamped_coul += float(line[25:])
                                elif line.startswith("CHARGE-QUADRUPOLE"):
                                    undamped_coul += float(line[25:])
                                elif line.startswith("CHARGE-OCTUPOLE"):
                                    undamped_coul += float(line[25:])
                                elif line.startswith("DIPOLE-DIPOLE"):
                                    undamped_coul += float(line[25:])
                                elif line.startswith("DIPOLE-QUADRUPOLE"):
                                    undamped_coul += float(line[25:])
                                elif line.startswith("QUADRUPOLE-QUADRUPOLE"):
                                    undamped_coul += float(line[25:])
                                    out_file.write(str(undamped_coul) + "\n")

#Extract all SAPT values into single csv file
def SAPT():
    output_file = output_directory + "SAPT.csv"
    with open(output_file, 'w') as out_file:
        out_file.write("File,SAPT Coulomb\n")
        input_file = input_directory + "Ground_truth_files/consolidated.txt"
        with open(input_file, 'r') as in_file:
            for line in in_file:
                line = line.strip()
                out_file.write(line[26:line.rfind("'")] + ",")
                splitted = line.split()
                out_file.write(splitted[4] + "\n")

#sort a file lexicographically
def sort_file_lex(filename, destination):
    df = pd.read_csv(filename)
    df = df.sort_values(by='File')
    df.to_csv(destination, index = False)

# Convert kcal/mol to hartree
# Current ground truth values from SAPT are in kcal/mol
def kcal_to_hartree(filename, column_name, destination):
    df = pd.read_csv(filename)
    df[column_name] = df[column_name].apply(lambda x: x * 0.00159362)
    df.to_csv(destination, index = False)

#Combine undamped Coulomb file and SAPT ground truth csvs into one file
#Load an SAPT dataframe, an undamped energy dataframe, and merge them.
def add_undamped_and_SAPT(sapt_source, undamped_source, destination):
    df1 = pd.read_csv(sapt_source)
    df2 = pd.read_csv(undamped_source)
    df_merged = pd.merge(df1, df2, on='File', how='left')

    #Drop rows missing entries of either kind
    df_merged = df_merged.dropna()

    #Save File
    df_merged.to_csv(destination, index = False)

#Make a JSON file out of a CSV file
def create_JSON(filename, destination):
    df = pd.read_csv(filename)
    metadata = {
        row["File"]: {
            "SAPT Coulomb": row["SAPT Coulomb"],
            "Undamped Coulomb": row["Undamped Coulomb"]
        }
        for _, row in df.iterrows()
    }
    with open(destination, "w") as f:
        json.dump(metadata, f, indent=4)
    
def calculate_errors(filename, destination):
    df = pd.read_csv(filename)
    df["Errors = |SAPT - Undamped|"] = abs(df["SAPT Coulomb"] - df["Undamped Coulomb"])
    df.drop(['SAPT Coulomb', 'Undamped Coulomb'], axis=1, inplace=True)
    df.to_csv(destination, index = False)


#Extract undamped energy for a single database
def undamped_extract(out_filename, input_directory):
    with open(out_filename, 'w') as out_file:
        out_file.write("File,Undamped Coulomb\n")
        for file_name in os.listdir(input_directory):
            if os.path.isfile(os.path.join(input_directory, file_name)):
                with open(input_directory + file_name, 'r') as in_file:
                    if file_name.endswith(".log"):
                        out_file.write(file_name[0:file_name.find("-unCP")] + ",")
                        undamped_coul = 0
                        for line in in_file:
                            line = line.strip()
                            if line.startswith("CHARGE-CHARGE"):
                                fields = line.split()
                                undamped_coul += float(fields[2])
                            elif line.startswith("CHARGE-DIPOLE"):
                                fields = line.split()
                                undamped_coul += float(fields[2])
                            elif line.startswith("CHARGE-QUADRUPOLE"):
                                fields = line.split()
                                undamped_coul += float(fields[2])
                            elif line.startswith("CHARGE-OCTUPOLE"):
                                fields = line.split()
                                undamped_coul += float(fields[2])
                            elif line.startswith("DIPOLE-DIPOLE"):
                                fields = line.split()
                                undamped_coul += float(fields[2])
                            elif line.startswith("DIPOLE-QUADRUPOLE"):
                                fields = line.split()
                                undamped_coul += float(fields[2])
                            elif line.startswith("QUADRUPOLE-QUADRUPOLE"):
                                fields = line.split()
                                undamped_coul += float(fields[2])
                                out_file.write(str(undamped_coul) + "\n")

#Extract SAPT energy for a single database
def SAPT_one_db(out_filename, in_filename, database):
    with open(out_filename, 'w') as out_file:
        out_file.write("File,SAPT Coulomb\n")
        with open(in_filename, 'r') as in_file:
            for line in in_file:
                line = line.strip()
                if "SAPT ELST ENERGY" in line and database in line:
                    out_file.write(line[line.find(database):line.rfind("'")] + ",")
                    splitted = line.split()
                    out_file.write(splitted[4] + "\n")  

# Used to combine undamped & SAPT for a new database or series of files 
# into a general master list of undamped & SAPT energies
def concatenate_vertically(master, new_addition, destination):
    df1 = pd.read_csv(master) #master list
    df2 = pd.read_csv(new_addition) #undamped/SAPT for single file
    df_merged = pd.concat([df1, df2], ignore_index=True)
    df_merged.to_csv(destination, index = False)                              

if __name__ == "__main__":
    #undamped()
    #SAPT()
    #sort_file_lex(output_directory + "SAPT.csv", output_directory + "SAPT.csv")
    #calculate_errors(output_directory + "meta.csv", output_directory + "errors.csv")
    #undamped_extract("/mnt/c/c++_tests/games_work/ml_work/clean_data/Ground_truth_files/S22by7/undampedS22by7_additional.csv", "/mnt/c/c++_tests/games_work/ml_work/Raw_data/S22by7_additional/")
    #SAPT_one_db("/mnt/c/c++_tests/games_work/ml_work/clean_data/Ground_truth_files/S22by7/SAPT_S22by7.csv","/mnt/c/c++_tests/games_work/ml_work/Raw_data/Ground_truth_files/sapt_S22by7.txt", "S22by7")
    #create_JSON("/mnt/c/c++_tests/games_work/ml_work/clean_data/Ground_truth_files/meta.csv", "/mnt/c/c++_tests/games_work/ml_work/clean_data/Ground_truth_files/meta.json")
    #calculate_errors("/mnt/c/c++_tests/games_work/ml_work/clean_data/Ground_truth_files/ACHC/SAPT_Undp_ACHC.csv", "/mnt/c/c++_tests/games_work/ml_work/clean_data/Ground_truth_files/ACHC/base_errors.csv")

    #concatenate_vertically("/mnt/c/c++_tests/games_work/ml_work/synth_data/70kplus750on90k_filtered01.csv", "/mnt/c/c++_tests/games_work/ml_work/MLModel/SymbolicReg/data/test1/all_damped_data.csv", "/mnt/c/c++_tests/games_work/ml_work/synth_data/Combined70750s90kr160kt.csv")
    #kcal_to_hartree("/mnt/c/c++_tests/games_work/ml_work/clean_data/Ground_truth_files/S22by7/SAPT_Undp_S22by7_additional.csv", "SAPT Coulomb", "/mnt/c/c++_tests/games_work/ml_work/clean_data/Ground_truth_files/S22by7/SAPT_Undp_S22by7_additional.csv")
    add_undamped_and_SAPT("/mnt/c/c++_tests/games_work/ml_work/MLModel/SymbolicReg/Trial_3/testdatax_predictions1.csv", "/mnt/c/c++_tests/games_work/ml_work/clean_data/Ground_truth_files/meta.csv", "/mnt/c/c++_tests/games_work/ml_work/MLModel/SymbolicReg/Trial_3/testdatax_predictions1.csv")
