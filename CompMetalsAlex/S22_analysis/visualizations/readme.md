# S22 Dataset Electrostatic Energy Visualizations

Here are some experimental visualizations, just done to explore whether there was any obvious structure to this data. The problem we face is that the functions we wish to learn depend on a series of variables $S_{ij}$ and $R_{ij}$. These are dependent on the distances and orbital overlaps molecules/atoms for each simulation file in the S22 dataset. Therefore, the amount of $S_{ij}$ and $R_{ij}$ values vary greatly from one output file to another. If we were to train a model on all of these values, we would end up with samples of different dimensionality, ranging from 32 to 882. This would require us to pad the smaller sample with zeroes. Another option would be to consider the sums of these $S_ij$ and $R_ij$ values and attempt to discover a relationship dependent on this sum. This would greatly reduce the dimensionality of the data, but the relevance of this from a chemistry point of view remains uncertain for now. I produced some visualizations to explore this, out of curiosity, which I present here as a means of record-keeping.

![Sij_vs_total_energy](sij_vs_total.png) 

![Rij_vs_total_energy](rij_vs_total.png)

![Sij_and_Rij_vs_total](rijandsij_vs_total.png)

![sijrijtotal_animated](rotating_scatter.gif)