import os
import pandas as pd
import numpy as np
import scipy
from scipy.optimize import curve_fit
import math
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter
import matplotlib
matplotlib.use('agg')

def log_fit(pair_data, plotname):
   # Load (Sij, Rij) data
   data = pd.read_csv(pair_data)

   # Extract variables
   Sij = data['Sij'].values
   Rij = data['Rij'].values

   # Define the model
   def log_model(s, a, b):
      return a * (-np.log(s)) + b

   # Fit the model to your data
   params, _ = curve_fit(log_model, Sij, Rij)

   a_fit, b_fit = params
   print(f"Fitted parameters: a = {a_fit:.4f}, b = {b_fit:.4f}")

   # Predict Rij values for plotting
   Sij_sorted = np.sort(Sij)
   Rij_fit = log_model(Sij_sorted, a_fit, b_fit)

   # Plot original data and fitted curve
   plt.figure(figsize=(20, 15))
   plt.scatter(Sij, Rij, alpha=0.2, label='Raw Data', s=10)
   plt.plot(Sij_sorted, Rij_fit, color='red', linewidth=2, label='Log Fit')
   
   plt.xlabel('Sij', fontsize=18)
   plt.ylabel('Rij', fontsize=18)
   plt.title(f'Rij vs Sij (90,000 samples)\n\nLogarithmic Fit: Rij = -{a_fit:.4f} ln(Sij) + {b_fit:.4f}', fontsize=18)
   plt.xticks(np.arange(0, 0.312, 0.013), fontsize=14)
   plt.yticks(np.arange(0, 25, 1), fontsize=14)
   plt.legend()
   plt.grid(True)
   plt.savefig(plotname)

if __name__ == "__main__":
   log_fit("/mnt/c/c++_tests/games_work/ml_work/MLModel/SymbolicReg/data/test1/all_damped_data.csv", "/mnt/c/c++_tests/games_work/ml_work/Graphs/logfit_sijvsrij90K.png")
