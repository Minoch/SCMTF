import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

def generate_phenotype_time_evolution(rank, factor2):
    # Generate time plot for each phenotype
    timepoints = [-3,-2,-1,0,1,2,3]
    for r in range(rank):
        values = factor2[:, r].detach().numpy()
        plt.plot(timepoints, values[:len(timepoints)])
        plt.title(f"Time evolution of phenotype {r + 1}")
        plt.xlabel("Timepoint")
        plt.xticks(timepoints)
        plt.ylabel("Value")
        plt.grid(True)
        plt.savefig(f"./model_diagnostic_images/time_evolution_phenotype_{r + 1}.png")
        plt.clf()
        plt.close()

    return