import numpy as np
import pandas as pd

def load_fitness_data(file_path):

    df = pd.read_excel(file_path)

    genotypes = df['genotype'].astype(str).values
    fitness_values = df['combined'].values
    
    fitness_table = np.zeros((2,) * 13) 
    
    for genotype, fitness in zip(genotypes, fitness_values):
        genotype = genotype.strip("'") 
        index = tuple(int(g) for g in genotype) 
        fitness_table[index] = fitness
    
    return fitness_table

file_path = '41467_2019_12130_MOESM7_ESM.xlsx'
fitness_table = load_fitness_data(file_path)
np.save('preprocessed_fitness_table.npy', fitness_table)


def get_fitness(genotype):
    index = tuple(int(g) for g in genotype)
    return fitness_table[index]

sample_genotype = '1100110100101'
fitness_value = get_fitness(sample_genotype)
print(f"Genotype {sample_genotype}: {fitness_value}")
