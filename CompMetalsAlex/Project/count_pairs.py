import pandas as pd
import os
import sympy

def count_pairs(input_directory):
   count = 0
   total_files = 0
   for file_name in os.listdir(input_directory):
      if os.path.isfile(os.path.join(input_directory, file_name)):
         df = pd.read_csv(input_directory + file_name, header=None, skiprows=1)
         count += len(df)
         total_files += 1
   print(f"Current file count with Sij/Rij info: {total_files}")
   print(f"The total amount of Sij/Rij pairs is: {count}")
   print(f"The average number of pairs per file is:{count/total_files}")

if __name__ == "__main__":
   count_pairs("/mnt/c/c++_tests/games_work/ml_work/clean_data/All_data/")