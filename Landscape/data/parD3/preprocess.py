import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.interpolate import LinearNDInterpolator


# Load the data from the provided image (the CSV content will be assumed to have been provided separately)
file_path = 'GSE153897_Variant_fitness.csv'

# Read CSV data
data = pd.read_csv(file_path)

# Prepare an empty (20, 20, 20, 2) ndarray and fill with NaN initially
amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
n = len(amino_acids)
fitness_array = np.full((n, n, n, 2), np.nan)

# Create a map from amino acids to index
aa_to_idx = {aa: i for i, aa in enumerate(amino_acids)}

# Iterate through the data and place the values into the ndarray
for _, row in data.iterrows():
    variant = row['Variant']
    fitness_E3 = row['W_ParE3']
    fitness_E2 = row['W_ParE2']
    
    idx1 = aa_to_idx[variant[0]]
    idx2 = aa_to_idx[variant[1]]
    idx3 = aa_to_idx[variant[2]]
    
    fitness_array[idx1, idx2, idx3, 0] = fitness_E3
    fitness_array[idx1, idx2, idx3, 1] = fitness_E2



# Generate grid for non-NaN values
valid_indices_E3 = np.argwhere(~np.isnan(fitness_array[:, :, :, 0]))
valid_fitness_E3 = fitness_array[:, :, :, 0][~np.isnan(fitness_array[:, :, :, 0])]
interp_func_E3 = LinearNDInterpolator(valid_indices_E3, valid_fitness_E3)
# Find NaN entries in ParE3 and interpolate those values
nan_indices_E3 = np.argwhere(np.isnan(fitness_array[:, :, :, 0]))
for idx in nan_indices_E3:
    interpolated_value = interp_func_E3(idx)
    fitness_array[idx[0], idx[1], idx[2], 0] = interpolated_value

# Interpolate NaN values for fitness_array[:,:,:,1] (ParE2 fitness) using the same logic
valid_indices_E2 = np.argwhere(~np.isnan(fitness_array[:, :, :, 1]))
valid_fitness_E2 = fitness_array[:, :, :, 1][~np.isnan(fitness_array[:, :, :, 1])]
interp_func_E2 = LinearNDInterpolator(valid_indices_E2, valid_fitness_E2)
# Find NaN entries in ParE2 and interpolate those values
nan_indices_E2 = np.argwhere(np.isnan(fitness_array[:, :, :, 1]))
for idx in nan_indices_E2:
    interpolated_value = interp_func_E2(idx)
    fitness_array[idx[0], idx[1], idx[2], 1] = interpolated_value


fitness_E3 = fitness_array[:, :, :, 0].flatten()
fitness_E2 = fitness_array[:, :, :, 1].flatten()

# Create a scatter3d plot for ParE3 fitness using colors to represent fitness levels
fig = go.Figure(data=[go.Scatter3d(
    x=np.tile(np.arange(20), 400), 
    y=np.tile(np.repeat(np.arange(20), 20), 20), 
    z=np.repeat(np.arange(20), 400), 
    mode='markers',
    marker=dict(
        size=5,
        color=fitness_E3,                # set color to the fitness values
        colorscale='Viridis',               # choose a colorscale
        colorbar=dict(title="Fitness"),     # add a color bar to show fitness levels
        opacity=0.8
    )
)])

# Add labels
fig.update_layout(
    title="3D Fitness Landscape for ParE3",
    scene=dict(
        xaxis_title="Amino Acid 1",
        yaxis_title="Amino Acid 2",
        zaxis_title="Amino Acid 3"
    ),
    margin=dict(l=0, r=0, b=0, t=40)
)

# Save the plot as an HTML file
fig.write_html("3d_fitness_landscape_ParE3.html")

# Save the interpolated fitness array as a numpy file
np.save("preprocessed_fitness_table.npy", fitness_array)