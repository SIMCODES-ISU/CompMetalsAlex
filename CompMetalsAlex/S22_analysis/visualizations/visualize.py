import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from matplotlib import cm
from mpl_toolkits import mplot3d
import matplotlib.animation as animation

directory = "/mnt/c/C++_TESTS/games_work/june_29_tests"

def sort_file():
   df = pd.read_csv(directory + "/visualizations/sumhartree.csv")
   df = df.sort_values(by='File', key=lambda s: s.str[4:].astype(int))
   df.to_csv("mod.csv", index = False)

def add_sapt():
   df = pd.read_csv(directory + "/visualizations/mod.csv")
   with open(directory + "/visualizations/sapt_S22.txt", 'r') as sapt:
         new_column_data = []
         for line in sapt:
               line = line.strip()
               new_column_data.append(float(line[line.rfind("-"):]) * 0.00159362)
         df["Ground Truth"] = new_column_data
         df.to_csv("modt.csv", index = False)

def graph_sij_vs_total():
   df = pd.read_csv(directory + "/visualizations/modt.csv")
   x_data = df['Total Sij']
   y_data = df['Ground Truth']
   plt.scatter(x_data, y_data)
   plt.xlabel('Total Sij')
   plt.ylabel('Electrostatic Energy')
   plt.xticks(np.arange(0,1.2,0.1))
   plt.yticks(np.arange(-0.052, 0, 0.005))
   plt.savefig('sij_vs_total.png')
   plt.show()

def graph_rij_vs_total():
   df = pd.read_csv(directory + "/visualizations/modt.csv")
   x_data = df[' Total Rij']
   y_data = df['Ground Truth']
   plt.scatter(x_data, y_data)
   plt.xlabel('Total Rij')
   plt.ylabel('Electrostatic Energy')
   plt.xticks(np.arange(89,2200,200))
   plt.yticks(np.arange(-0.052, 0, 0.005))
   plt.savefig('rij_vs_total.png')
   plt.show()

def update(frame):
        # Example: Rotate the view
        ax.view_init(elev=20, azim=frame * 2) # Adjust rotation speed as needed
        return scatter,

def graph_rij_and_sij_vs_total():
   df = pd.read_csv(directory + "/visualizations/modt.csv")
   fig = plt.figure()
   ax = plt.axes(projection='3d')
   x_data = df[' Total Rij']
   y_data = df['Total Sij']
   z_data = df['Ground Truth']
   ax.set_xlabel(' Total Rij')
   ax.set_ylabel('Total Sij')
   ax.set_zlabel('Total Electrostatic')
   scatter = ax.scatter(x_data, y_data, z_data)
   plt.savefig('rijandsij_vs_total.png')
   plt.show()


   
   



if __name__ == "__main__":
   graph_rij_vs_total()
   graph_sij_vs_total()
   graph_rij_and_sij_vs_total()
   # df = pd.read_csv(directory + "/visualizations/modt.csv")
   # fig = plt.figure()
   # ax = plt.axes(projection='3d')
   # x_data = df[' Total Rij']
   # y_data = df['Total Sij']
   # z_data = df['Ground Truth']
   # ax.set_xlabel(' Total Rij')
   # ax.set_ylabel('Total Sij')
   # ax.set_zlabel('Total Electrostatic')
   # scatter = ax.scatter(x_data, y_data, z_data)

   # ani = animation.FuncAnimation(fig, update, frames=range(0, 360 // 4), interval=50, blit=True)
   # ani.save('rotating_scatter.gif', writer='pillow', fps=20)


