import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
import torch

#### For reference ####
# factor0 samples x rank (common matrix), indicated patient membership in phenotype
# factor1 temporal features x rank
# factor2 timepoints x rank
# mfactor1 static features x rank
#######################

def get_feature_mappings(static_list, lab_list, PRO_list):
    # Get feature mappings
    static_mapping = {}
    for x in range(len(static_list)):
        static_mapping[static_list[x]] = x

    temporal_mapping = {}
    temporal_list = lab_list + PRO_list # order matters
    for y in range(len(temporal_list)):
        temporal_mapping[temporal_list[y]] = y

    return static_mapping, temporal_mapping

def plot_phenotype_importance_scores(phenotypes, temporal, important_indices):
    selected_phenotypes = {key: phenotypes[key] for idx, key in enumerate(phenotypes) if idx in important_indices}
    print(important_indices)

    for phenotype_index, (phenotype_key, phenotype_data) in enumerate(selected_phenotypes.items()):
        if temporal:
            print(f"Plotting the temporal phenotype importance scores for the one-indexed phenotype {phenotype_key}")

            title = f"Contribution Score by Temporal Features for {phenotype_key}"
            xlabel = "Temporal Feature"
            filename = f"./final_phenotype_images/temporal_features_contribution_score_{phenotype_key.replace(' ', '_').lower()}.png"
            color = '#5DADE2'
        else:
            print(f"Plotting the static phenotype importance scores for the one-indexed phenotype {phenotype_key}")

            title = f"Contribution Score by Static Features for {phenotype_key}" # one-indexed
            xlabel = "Static Feature"
            filename = f"./final_phenotype_images/static_features_contribution_score_{phenotype_key.replace(' ', '_').lower()}.png"
            color = '#8ABA96'

        phenotype_array = np.array(phenotype_data, dtype=object)
        features = phenotype_array[:, 0]  
        scores = phenotype_array[:, 1].astype(float)

        plt.figure(figsize=(12, 7))
        plt.bar(features, scores, color=color)
        plt.xlabel(xlabel)
        plt.ylabel('Contribution Score')
        plt.title(title)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(filename)
        plt.clf()
        plt.close()

def extract_phenotypes(phenotype_matrix, mapping):

    index_to_feature_name = {value: key for key, value in mapping.items()}
    phenotypes = {}

    # Loop through each feature for the phenotype
    for col in range(phenotype_matrix.size(1)):
        column_values = phenotype_matrix[:, col]

        phenotype_data = [] # Use a list to store phenotype data

        for row_index in range(column_values.size(0)):
            value = column_values[row_index]
            dictionary_key = index_to_feature_name.get(row_index) 
            if dictionary_key:
                phenotype_data.append((dictionary_key, round(value.item(), 3)))
                # Append a tuple containing the feature name and its rounded value to the phenotype data list.
            else:
                print(f"Index {row_index} not found in dictionary")

        # Sort the phenotypes in descending order by value
        phenotype_data.sort(key=lambda item: item[1], reverse=True)

        # Store the sorted data for each phenotype
        phenotypes[f'Phenotype {col + 1}'] = phenotype_data

    return phenotypes

def create_splits(labels2, labels3, device, seed=1221):
    # Combine labels into a single array for stratification
    combined_labels = [f"{l1}_{l2}" for l1, l2 in zip(labels2, labels3)]

    # Create indices for the full dataset
    all_indices = np.arange(len(combined_labels))

    # First split: Train and Temp (which will be split again into val and test)
    train_idx, temp_idx, _, temp_labels = train_test_split(
        all_indices,
        combined_labels,
        test_size=0.4,  # 40% goes to temp, 60% to train
        random_state=seed,
        stratify=combined_labels
    )

    # Second split: Temp into Validation and Test
    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=0.5,  # Split temp into equal parts: 20% val, 20% test
        random_state=seed,
        stratify=temp_labels
    )

    # Convert to tensor
    train_idx = torch.tensor(train_idx).to(device)
    val_idx = torch.tensor(val_idx).to(device)
    test_idx = torch.tensor(test_idx).to(device)

    return train_idx, val_idx, test_idx

def avg_phenotypes_per_pt(factor0):
    # Convert the tensor to a numpy array for easier percentile calculations
    factor0_np = factor0.clone().detach().numpy()
    
    # Determine the number of phenotypes (columns)
    num_phenotypes = factor0_np.shape[1]

    # Iterate over each phenotype to calculate the 75th percentile and update values
    for i in range(num_phenotypes):
        # Calculate the 75th percentile for the current phenotype
        percentile_75_value = np.percentile(factor0_np[:, i], 75)

        # Update the tensor values based on the 75th percentile
        factor0_np[:, i] = torch.tensor(factor0_np[:, i] >= percentile_75_value, dtype=torch.float32)

    # Calculate the average number of phenotypes each patient belongs to
    avg_phenotypes_per_patient = factor0_np.sum(axis=1).mean().item()

    return avg_phenotypes_per_patient
