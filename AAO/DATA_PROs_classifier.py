import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import roc_auc_score, roc_curve, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt

def RF_classifier(factor0, labels, train_idx, val_idx, test_idx, importance_threshold=0.05, final_phenotype_indices=None):
    x_matrix = factor0.detach().numpy()
    
    if final_phenotype_indices:
        print(final_phenotype_indices)
        x_matrix = x_matrix[:, final_phenotype_indices]

    # Convert tensors to numpy arrays for indexing
    train_idx = train_idx.cpu().numpy()
    val_idx = val_idx.cpu().numpy()
    test_idx = test_idx.cpu().numpy()

    # Use both training and validation sets for training here
    train_idx = np.concatenate((train_idx, val_idx))

    # Index the x_matrix and labels using train and test indices
    x_train = x_matrix[train_idx]
    x_test = x_matrix[test_idx]
    y_train = labels[train_idx]
    y_test = labels[test_idx]

    print(x_train.shape)
    print(x_test.shape)
    print(y_test.shape)

    # Initialize Random Forest Classifiers for each outcome
    rf1 = RandomForestClassifier(random_state=42)
    rf2 = RandomForestClassifier(random_state=42)

    # Fit models
    rf1.fit(x_train, y_train[:, 0])  # Train for first outcome
    rf2.fit(x_train, y_train[:, 1])  # Train for second outcome

    # Predict probabilities
    y_pred_proba_1 = rf1.predict_proba(x_test)[:, 1]
    y_pred_proba_2 = rf2.predict_proba(x_test)[:, 1]

    # Calculate AUCs
    auc_1 = roc_auc_score(y_test[:, 0], y_pred_proba_1)
    auc_2 = roc_auc_score(y_test[:, 1], y_pred_proba_2)

    threshold = 0.5
    y_pred_1 = (y_pred_proba_1 >= threshold).astype(int)
    y_pred_2 = (y_pred_proba_2 >= threshold).astype(int)

    # Calculate precision, recall, and F1 score
    precision_1 = precision_score(y_test[:, 0], y_pred_1)
    recall_1 = recall_score(y_test[:, 0], y_pred_1)
    f1_1 = f1_score(y_test[:, 0], y_pred_1)

    precision_2 = precision_score(y_test[:, 1], y_pred_2)
    recall_2 = recall_score(y_test[:, 1], y_pred_2)
    f1_2 = f1_score(y_test[:, 1], y_pred_2)

    # Print metrics
    print(f"AUC for yr2 outcome: {auc_1:.4f}")
    print(f"Precision for yr2 outcome: {precision_1:.4f}")
    print(f"Recall for yr2 outcome: {recall_1:.4f}")
    print(f"F1 score for yr2 outcome: {f1_1:.4f}")

    print(f"AUC for yr3 outcome: {auc_2:.4f}")
    print(f"Precision for yr3 outcome: {precision_2:.4f}")
    print(f"Recall for yr3 outcome: {recall_2:.4f}")
    print(f"F1 score for yr3 outcome: {f1_2:.4f}")

    # Plot
    print(f"AUC for yr2 outcome: {auc_1:.4f}")
    print(f"AUC for yr3 outcome: {auc_2:.4f}")

    # Feature Importances
    importances_1 = rf1.feature_importances_
    importances_2 = rf2.feature_importances_

    # Printing feature importances in a readable format
    features = [f"Phenotype {i + 1}" for i in range(x_matrix.shape[1])] # one-indexed
    importance_df_1 = pd.DataFrame({'Phenotype': features, 'Importance': importances_1})
    importance_df_2 = pd.DataFrame({'Phenotype': features, 'Importance': importances_2})

    print("Phenotype importances for yr2 outcome:")
    print(importance_df_1.sort_values(by='Importance', ascending=False))

    print("Phenotype importances for yr3 outcome:")
    print(importance_df_2.sort_values(by='Importance', ascending=False))

    # Indices of features with importance >= 0.05
    important_indices_1 = [i for i, importance in enumerate(importances_1) if importance >= importance_threshold]
    important_indices_2 = [i for i, importance in enumerate(importances_2) if importance >= importance_threshold]

    print("RF1")
    print(rf1)
    
    print("RF2")
    print(rf2)

    return important_indices_1, important_indices_2, importance_df_1, importance_df_2 # results are zero-indexed

# Plot RF feature importance
def plot_rf_feature_importances(importances_df, title=None, filename=None):
    # Sort the DataFrame by importance in descending order
    sorted_df = importances_df.sort_values(by='Importance', ascending=False)
    
    # Extract sorted feature names and importances
    features = sorted_df['Phenotype']
    importances = sorted_df['Importance']

    # Plotting
    color='#B37448'
    plt.figure(figsize=(12, 7))
    plt.bar(features, importances, color=color)

    # Adding titles and labels
    plt.xlabel("Features")
    plt.ylabel("Importance")
    plt.title(title)

    # Adjust x-ticks
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    # Save and close the plot
    plt.savefig(filename)
    plt.clf()
    plt.close()