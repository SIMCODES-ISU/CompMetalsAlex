import os
import math
import pandas as pd

def damping(Sij, Rij):
    return -2 * math.sqrt(1 / (-2 * math.log(Sij))) * (Sij*Sij / Rij)

all_rows = []



def extract_and_damp(input_directory, output_directory):
   for filename in os.listdir(input_directory):
      if filename.endswith(".csv"):
         df = pd.read_csv(os.path.join(input_directory, filename))
         for sij, rij in zip(df["Sij"], df["Rij"]):
               if sij > 0 and rij > 0:
                  value = damping(sij, rij)
                  all_rows.append({"Sij": sij, "Rij": rij, "damping_value": value})

   # Limit to 1000 for proof of concept
   pd.DataFrame(all_rows).to_csv(output_directory + "all_damped_data.csv", index=False)
   pd.DataFrame(all_rows[:1000]).to_csv(output_directory + "small_test_1000.csv", index=False)

if __name__ == "__main__":
   extract_and_damp("/mnt/c/c++_tests/games_work/ml_work/clean_data/All_data/", "/mnt/c/c++_tests/games_work/ml_work/MLModel/SymbolicReg/data/test1/")