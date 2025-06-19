import pandas as pd
import numpy as np
import matplotlib.pyplot as plt



def set_cpu_num(cpu_num: int = 1):
    import os
    if cpu_num <= 0: return
    
    os.environ ['OMP_NUM_THREADS'] = str(cpu_num)
    os.environ ['OPENBLAS_NUM_THREADS'] = str(cpu_num)
    os.environ ['MKL_NUM_THREADS'] = str(cpu_num)
    os.environ ['VECLIB_MAXIMUM_THREADS'] = str(cpu_num)
    os.environ ['NUMEXPR_NUM_THREADS'] = str(cpu_num)
set_cpu_num(64)



# Load the Excel file
file_path = '41586_2018_170_MOESM2_ESM.xlsx'
xls = pd.ExcelFile(file_path)

# Load the sheet
data = pd.read_excel(xls, sheet_name='Sheet1')

# Clean the data by skipping the first few header rows and resetting the index
data_cleaned = data.iloc[4:].reset_index(drop=True)

# Rename columns for ease of access
data_cleaned.columns = ['seq', 'num_vars', 'pos_vars', 'ntd_vars', 'ntd_wt', 'reads_IN_1', 'reads_IN_2', 
                        'reads_OUT_11', 'reads_OUT_12', 'reads_OUT_13', 'reads_OUT_21', 'reads_OUT_22', 
                        'reads_OUT_23', 'id', 'fitness', 'SE']

# Define the positions of interest (mutation positions)
mutation_positions = [1, 2, 6, 27, 43, 46, 66, 69, 70, 71]  # example positions (1-based)

# Function to extract the specific mutation positions from the sequence
def extract_mutation_positions(seq, positions):
    # Convert 1-based positions to 0-based for python indexing
    positions = [p - 1 for p in positions]
    return ''.join([seq[p] for p in positions])

# Apply the extraction of the 10 mutation positions for all valid sequences
full_sequences = data_cleaned['seq'].iloc[1:]  # Skip the first row which is a header
mutated_sequences = full_sequences.apply(lambda x: extract_mutation_positions(x, mutation_positions))

# Convert fitness and SE columns into a numpy array
fitness_se_array = data_cleaned[['fitness', 'SE']].iloc[1:].to_numpy()

# Convert the mutated sequences into a numpy string array
mutated_sequences_array = mutated_sequences.to_numpy()

# M<utation encoding
# Map the nucleotides to categorical values
nucleotide_to_index = {'A': 0, 'U': 1, 'G': 2, 'C': 3}
# Function to convert a sequence of 10 nucleotides into a categorical array (4176x10x1)
def nucleotide_to_categorical(seq):
    return np.array([nucleotide_to_index[nt] for nt in seq])
# Apply the function to all mutated sequences
categorical_encoding = np.array([nucleotide_to_categorical(seq) for seq in mutated_sequences])


# Output the shapes of the arrays to verify
print(f"Fitness and SE array shape: {fitness_se_array.shape}")
print(f"Mutated sequences array shape: {mutated_sequences_array.shape}")
print(f"Categorical encoding shape: {categorical_encoding.shape}")




# import seaborn as sns
# # First, let's calculate the frequency of each nucleotide (A=0, C=1, G=2, U=3) at each of the 10 positions
# nucleotide_counts = np.zeros((10, 4))  # 10 positions, 4 nucleotides
# # Loop over each position (10) and count nucleotide occurrences
# for i in range(10):
#     nucleotide_counts[i] = np.bincount(categorical_encoding[:, i], minlength=4)
# # Normalize to get proportions
# nucleotide_proportions = nucleotide_counts / np.sum(nucleotide_counts, axis=1, keepdims=True)
# # Stacked bar chart
# def plot_stacked_bar_chart():
#     nucleotides = ['A', 'U', 'G', 'C']
#     positions = np.arange(1, 11)
#     # Use the 'RdYlGn' colormap to match the color scheme in the screenshot
#     category_colors = plt.get_cmap('RdYlGn')(np.linspace(0.15, 0.75, 4))
#     # Plot the stacked bar chart
#     fig, ax = plt.subplots(figsize=(7, 4))
#     bottom = np.zeros(10)
#     for i in range(4):  # For each nucleotide (A, C, G, U)
#         ax.bar(positions, nucleotide_proportions[:, i], bottom=bottom, label=nucleotides[i], color=category_colors[i])
#         bottom += nucleotide_proportions[:, i]
#     ax.set_xlabel('Mutation Position')
#     ax.set_ylabel('Proportion')
#     ax.set_xticks(positions)
#     ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.125), ncol=4, frameon=False, fontsize=12)
#     plt.show()
# plot_stacked_bar_chart()




# Fitness table
map = {0:{0:0,2:1}, 1:{1:0,2:1,3:2}, 2:{0:0,1:1,2:2}, 3:{1:0,3:1}, 4:{0:0,3:1}, 5:{1:0,3:1}, 6:{0:0,1:1,3:2}, 7:{0:0,2:1}, 8:{0:0,1:1,2:2}, 9:{1:0,3:1}}
fitness_table = np.zeros((2,3,3,2,2,2,3,2,3,2))
fitness_table.fill(np.nan)
for idx, sample in enumerate(categorical_encoding):
    fitness = fitness_se_array[idx, 0]
    index = [map[0][sample[0]], map[1][sample[1]], map[2][sample[2]], map[3][sample[3]], map[4][sample[4]], map[5][sample[5]], map[6][sample[6]], map[7][sample[7]], map[8][sample[8]], map[9][sample[9]]]
    # fitness_table[*index] = fitness
    fitness_table[index[0], index[1], index[2], index[3], index[4], index[5], index[6], index[7], index[8], index[9]] = fitness


# # Generate grid for non-NaN values
# valid_indices = np.argwhere(~np.isnan(fitness_table))
# valid_fitness = fitness_table[~np.isnan(fitness_table)]
# # interp_func = LinearNDInterpolator(valid_indices, valid_fitness)
# interp_func = NearestNDInterpolator(valid_indices, valid_fitness)
# # Find NaN entries in ParE3 and interpolate those values
# nan_indices = np.argwhere(np.isnan(fitness_table))
# for idx in nan_indices:
#     interpolated_value = interp_func(idx)
#     fitness_table[*idx] = interpolated_value




from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# Generate features (input data) and labels (target data) for the model
valid_indices = np.argwhere(~np.isnan(fitness_table))
valid_fitness = fitness_table[~np.isnan(fitness_table)]
# Split the data into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(valid_indices, valid_fitness, test_size=0.2, random_state=42)

# For Linear Regression:
from sklearn.linear_model import LinearRegression
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
y_pred_lr = lr_model.predict(X_test)
r2_lr = r2_score(y_test, y_pred_lr)
mae_lr = mean_absolute_error(y_test, y_pred_lr)
nmse_lr = (mean_squared_error(y_test, y_pred_lr, multioutput='raw_values') / np.var(y_test, keepdims=True)).mean()
print(f'LR R2: {r2_lr:.4f}, MAE: {mae_lr:.4f}, NMSE: {nmse_lr:.4f}')

# For SVR:
from sklearn.svm import SVR
svr_model = SVR(kernel='rbf', C=100, epsilon=0.1)
svr_model.fit(X_train, y_train)
y_pred_svr = svr_model.predict(X_test)
r2_svr = r2_score(y_test, y_pred_svr)
mae_svr = mean_absolute_error(y_test, y_pred_svr)
nmse_svr = (mean_squared_error(y_test, y_pred_svr, multioutput='raw_values') / np.var(y_test, keepdims=True)).mean()
print(f'SVR R2: {r2_svr:.4f}, MAE: {mae_svr:.4f}, NMSE: {nmse_svr:.4f}')

# For Random Forest:
from sklearn.ensemble import RandomForestRegressor
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)
r2_rf = r2_score(y_test, y_pred_rf)
mae_rf = mean_absolute_error(y_test, y_pred_rf)
nmse_rf = (mean_squared_error(y_test, y_pred_rf, multioutput='raw_values') / np.var(y_test, keepdims=True)).mean()
print(f'RF R2: {r2_rf:.4f}, MAE: {mae_rf:.4f}, NMSE: {nmse_rf:.4f}')

# For XGBoost:
import xgboost as xgb
xgb_model = xgb.XGBRegressor(n_estimators=100, random_state=42)
xgb_model.fit(X_train, y_train)
y_pred_xgb = xgb_model.predict(X_test)
r2_xgb = r2_score(y_test, y_pred_xgb)
mae_xgb = mean_absolute_error(y_test, y_pred_xgb)
nmse_xgb = (mean_squared_error(y_test, y_pred_xgb, multioutput='raw_values') / np.var(y_test, keepdims=True)).mean()
print(f'XGB R2: {r2_xgb:.4f}, MAE: {mae_xgb:.4f}, NMSE: {nmse_xgb:.4f}')

# For LightGBM:
import lightgbm as lgb
lgb_model = lgb.LGBMRegressor(n_estimators=100, random_state=42, verbose=-1, force_row_wise=True)
lgb_model.fit(X_train, y_train)
y_pred_lgb = lgb_model.predict(X_test)
r2_lgb = r2_score(y_test, y_pred_lgb)
mae_lgb = mean_absolute_error(y_test, y_pred_lgb)
nmse_lgb = (mean_squared_error(y_test, y_pred_lgb, multioutput='raw_values') / np.var(y_test, keepdims=True)).mean()
print(f'LGB R2: {r2_lgb:.4f}, MAE: {mae_lgb:.4f}, NMSE: {nmse_lgb:.4f}')

# For Gaussian Process:
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
gp_model = GaussianProcessRegressor(kernel=RBF() + WhiteKernel(), random_state=42)
gp_model.fit(X_train, y_train)
y_pred_gp = gp_model.predict(X_test)
r2_gp = r2_score(y_test, y_pred_gp)
mae_gp = mean_absolute_error(y_test, y_pred_gp)
nmse_gp = (mean_squared_error(y_test, y_pred_gp, multioutput='raw_values') / np.var(y_test, keepdims=True)).mean()
print(f'GP R2: {r2_gp:.4f}, MAE: {mae_gp:.4f}, NMSE: {nmse_gp:.4f}')

# For Neural Network:
from sklearn.neural_network import MLPRegressor
nn_model = MLPRegressor(hidden_layer_sizes=(32, 32), max_iter=1000, random_state=42)
nn_model.fit(X_train, y_train)
y_pred_nn = nn_model.predict(X_test)
r2_nn = r2_score(y_test, y_pred_nn)
mae_nn = mean_absolute_error(y_test, y_pred_nn)
nmse_nn = (mean_squared_error(y_test, y_pred_nn, multioutput='raw_values') / np.var(y_test, keepdims=True)).mean()
print(f'NN R2: {r2_nn:.4f}, MAE: {mae_nn:.4f}, NMSE: {nmse_nn:.4f}')

# For Nearest Neighbors:
from sklearn.neighbors import KNeighborsRegressor
knn_model = KNeighborsRegressor(n_neighbors=5)
knn_model.fit(X_train, y_train)
y_pred_knn = knn_model.predict(X_test)
r2_knn = r2_score(y_test, y_pred_knn)
mae_knn = mean_absolute_error(y_test, y_pred_knn)
nmse_knn = (mean_squared_error(y_test, y_pred_knn, multioutput='raw_values') / np.var(y_test, keepdims=True)).mean()
print(f'KNN R2: {r2_knn:.4f}, MAE: {mae_knn:.4f}, NMSE: {nmse_knn:.4f}')

# Prepare data for NaN entries (to be predicted)
nan_indices = np.argwhere(np.isnan(fitness_table))
nan_predictions_lgb = lgb_model.predict(nan_indices)
for i, idx in enumerate(nan_indices):
    fitness_table[tuple(idx)] = nan_predictions_lgb[i]

# Save the preprocessed data
np.save('./preprocessed_fitness_table.npy', fitness_table)