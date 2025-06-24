import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import proximal_gradient.proximalGradient as pg

import tensorly as tl
from tensorly.cp_tensor import cp_to_tensor

from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score

def weights_init(layer_in):
        if isinstance(layer_in, nn.Linear):
            nn.init.kaiming_normal_(layer_in.weight)
            layer_in.bias.data.fill_(0.0)
        return
    
class PatientSlicesTensor(nn.Module):
    def __init__(self, depth, height, width):
        super().__init__()
        # One trainable scalar per slice
        self.slice_values = nn.Parameter(torch.rand(depth))
        self.height = height
        self.width = width
    def clamp_it(self):
        return torch.clamp(self.slice_values.data, min=-1, max=1).data
    def forward(self):
        # Expand each scalar to an HxW slice
        return self.slice_values[:, None, None].expand(-1, self.height, self.width)

class FeatureSlicesTensor(nn.Module):
    def __init__(self, depth, height, width):
        super().__init__()
        # One trainable scalar per slice
        self.depth = depth
        self.slice_values = nn.Parameter(torch.rand(height))
        self.width = width
    def clamp_it(self):
        return torch.clamp(self.slice_values.data, min=-1, max=1).data
    def forward(self):
        return self.slice_values[None, :, None].expand(self.depth, -1, self.width)


class JointDecompositionModel(nn.Module):
    def __init__(self, rank, n_epoch, Lambda, lr, penalty_l1, penalty_l2, theta, tensor_, Mtensor_, rng, device):

        self.rank = rank
        self.n_epoch = n_epoch
        self.Lambda = Lambda
        self.lr = lr
        self.penalty_l1 = penalty_l1
        self.penalty_l2 = penalty_l2
        self.theta = theta

        super(JointDecompositionModel, self).__init__() 

        # Initialize factors with requires_grad=True for backpropagation
        self.factor0 = nn.Parameter(tl.tensor(rng.random_sample((tensor_.shape[0], rank)), device=device), requires_grad=True)  # samples x rank (common matrix)
        self.factor1 = nn.Parameter(tl.tensor(rng.random_sample((tensor_.shape[1], rank)), device=device), requires_grad=True)  # temporal features x rank
        self.factor2 = nn.Parameter(tl.tensor(rng.random_sample((tensor_.shape[2], rank)), device=device), requires_grad=True)  # timepoints x rank
        self.Mfactor1 = nn.Parameter(tl.tensor(rng.random_sample((Mtensor_.shape[1], rank)), device=device), requires_grad=True)  # static features x rank
        self.weights = nn.Parameter(torch.ones(rank, device=device), requires_grad=True) # Initialized to ones

        # Define the linear layers and ReLU using nn.Sequential for the classifier
        D_i, D_k, D_o = rank, 10, 2
        self.classifier = nn.Sequential(
            nn.Linear(D_i, D_k),
            nn.ReLU(),
            nn.BatchNorm1d(D_k),
            nn.Linear(D_k, D_o),
        )

        # Apply the weights initialization
        self.classifier.apply(weights_init)  

    def normalize(self, tensor_):
        # Normalize tensor column-wise using L2 norm 
        norm = torch.norm(tensor_, p=2, dim=0, keepdim=True) # vector of norms, one for each column
        # Avoid division by zero
        epsilon = 1e-8
        norm = torch.clamp(norm, min=epsilon)
        return tensor_ / norm, norm 
    
    def clamp(self, tensor_):
        # Clamp using ReLu
        return F.relu(tensor_)

    def forward(self, samples):
        return self.classifier(self.factor0[samples])

    def clamp_post_backprop(self):

        # Apply L2 normalization to each factor
        normalized_factor0, norm0 = self.normalize(self.factor0)
        normalized_factor1, norm1 = self.normalize(self.factor1)
        normalized_factor2, norm2 = self.normalize(self.factor2)
        normalized_Mfactor1, norm3 = self.normalize(self.Mfactor1)

        # Update the weights using the product of norms
        norms_product = torch.mul(norm0,norm1)
        norms_product = torch.mul(norms_product,norm2)
        norms_product = torch.mul(norms_product,norm3)
        self.weights.data = norms_product.squeeze() 

        # Update factors without detaching gradients
        self.factor0.data = self.clamp(normalized_factor0).data
        self.factor1.data = self.clamp(normalized_factor1).data
        self.factor2.data = self.clamp(normalized_factor2).data
        self.Mfactor1.data = self.clamp(normalized_Mfactor1).data
    
# Create Masks
def create_masks(tensor_, device, drop_prob=0.05):
    mask1 = ~torch.isnan(tensor_)
    mask2 = torch.FloatTensor(*tensor_.shape).uniform_() > drop_prob
    mask2 = mask2.to(device)
    mask12 = mask1 * mask2
    mask1, mask12 = mask1.to(device), mask12.to(device)
    return mask1, mask12

# Replace NaNs with 0
def nan_to_zero(tensor_):
        tensor_[tensor_ != tensor_] = 0  

    
def run_model(tensor, tensor1, Mtensor, 
              rng, device, 
              train_idx, test_idx, val_idx, labels, 
              rank, n_epoch, Lambda, lr, 
              penalty_l1, penalty_l2, theta, grid=False):
    relu = nn.ReLU()
    sig = nn.Sigmoid()

    mask1, mask12 = create_masks(tensor, device)
    nan_to_zero(tensor)
    nan_to_zero(tensor1)

    model = JointDecompositionModel(rank=rank, tensor_=tensor, Mtensor_=Mtensor, rng=rng, device=device, theta=theta, n_epoch=n_epoch, Lambda=Lambda, lr=lr, penalty_l1=penalty_l1, penalty_l2=penalty_l2).to(device)

    # Patient bias
    patient_uniform = PatientSlicesTensor(depth=model.factor0.shape[0], height=model.factor1.shape[0], width=model.factor2.shape[0]).to(device)
    patient_bias = patient_uniform()

    # Temporal feature bias
    feature_uniform = FeatureSlicesTensor(depth=model.factor0.shape[0], height=model.factor1.shape[0], width=model.factor2.shape[0]).to(device)
    feature_bias = feature_uniform()

    optimizer1 = torch.optim.Adam([model.weights, model.factor0, model.factor1, model.factor2, model.Mfactor1, 
                                        patient_uniform.slice_values, feature_uniform.slice_values], lr=lr) # already has momentum
    optimizer2 = torch.optim.SGD(model.classifier.parameters(), lr=0.001, momentum=0.9)

    # Add learning rate schedulers
    scheduler1 = optim.lr_scheduler.StepLR(optimizer1, step_size=1000, gamma=0.8)
    scheduler2 = optim.lr_scheduler.StepLR(optimizer2, step_size=1000, gamma=0.8)

    # mask1.sum()/(shape[0]*shape[1]*shape[2]) # Fraction of values masked with mask 1
    # mask12.sum()/(shape[0]*shape[1]*shape[2])

    criterion = nn.BCEWithLogitsLoss()

    # Training Loop
    diagnostic_plot_reconstruction_loss_vals = []

    for i in range(1, n_epoch):

        optimizer1.zero_grad()
        optimizer2.zero_grad()
        
        # Reconstruct the tensor from the decomposed form
        rec = relu(cp_to_tensor((model.weights, [model.factor0,model.factor1,model.factor2]))) + patient_bias + feature_bias
        Mrec = relu(cp_to_tensor((model.weights, [model.factor0,model.Mfactor1]))) 
        
        # Calculate reconstruction loss
        Tloss = tl.norm(mask12 * (rec - tensor), 2)
        Mloss = tl.norm(Mrec - Mtensor, 2)

        # Apply regularization
        loss = Tloss + Mloss
        for f in [model.factor0, model.factor1, model.Mfactor1]:
            loss += penalty_l2 * f.pow(2).sum() + penalty_l1 * torch.sum(torch.abs(f))

        # Forward pass
        pred = model(train_idx)
        patient_bias = patient_uniform() 
        feature_bias = feature_uniform()

        # Classifier operation
        loss_cl = criterion(pred, labels[train_idx])
        loss_tot = (1 - Lambda) * loss + Lambda * loss_cl

        # Backpropagation the gradients and optimization step
        loss_tot.backward() # compute the gradients 

        # Clamp after backpropagation
        model.clamp_post_backprop()
        feature_uniform.clamp_it()
        patient_uniform.clamp_it()

        optimizer1.step() # apply the gradients 
        optimizer2.step()

        scheduler1.step()
        scheduler2.step()

        pg.l1(model.factor0, reg=0.005)
        pg.l1(model.factor1, reg=0.005)
        pg.l1(model.factor2, reg=0.005)
        pg.l1(model.Mfactor1, reg=0.005)

        if i % 100 == 0:
            rec_error = (tl.norm(mask12*(rec.data - tensor.data), 2) + 
                            tl.norm(Mrec.data - Mtensor.data, 2))/(tl.norm(mask12*tensor.data, 2) + tl.norm(Mtensor.data, 2))
            diagnostic_plot_reconstruction_loss_vals = np.append(diagnostic_plot_reconstruction_loss_vals, rec_error)

        if i == n_epoch - 1:
            print(model.weights)

    ### Evaluate model on test set
    factors = [model.factor0, model.factor1, model.factor2]
    rec = relu(cp_to_tensor((model.weights, factors))) + patient_bias + feature_bias
    tl.norm(((~mask12)*mask1)*(rec - tensor1), 2)/tl.norm(((~mask12)*mask1)*tensor1, 2)

    # Predictions
    if grid:
        final_idx = val_idx
    else:
        final_idx = test_idx
    pred = model(final_idx)

    # Metrics
    MAE_val = ((~mask12)*mask1*(rec - tensor1)).abs().sum()/((~mask12)*mask1).sum() # MAE imputation
    RMSE_val = torch.sqrt(((~mask12)*mask1*(rec - tensor1)).square().sum()/((~mask12)*mask1).sum()) # RMSE imputation
    # print("length of predictions: ", len(pred))
    outcome2_auc = roc_auc_score(labels[final_idx][:,0].cpu().detach().numpy(), pred[:,0].cpu().detach().numpy())
    outcome3_auc = roc_auc_score(labels[final_idx][:,1].cpu().detach().numpy(), pred[:,1].cpu().detach().numpy())

    # Additional metrics
    outcome2_pred_labels = (pred[:,0] > 0.5).cpu().detach().numpy()
    outcome2_true_labels = labels[final_idx][:,0].cpu().detach().numpy()

    # Outcome 3
    outcome3_pred_labels = (pred[:,1] > 0.5).cpu().detach().numpy()
    outcome3_true_labels = labels[final_idx][:,1].cpu().detach().numpy()

    # Calculate precision, recall, and F1 score for outcome 2
    outcome2_precision = precision_score(outcome2_true_labels, outcome2_pred_labels)
    outcome2_recall = recall_score(outcome2_true_labels, outcome2_pred_labels)
    outcome2_f1 = f1_score(outcome2_true_labels, outcome2_pred_labels)

    # Calculate precision, recall, and F1 score for outcome 3
    outcome3_precision = precision_score(outcome3_true_labels, outcome3_pred_labels)
    outcome3_recall = recall_score(outcome3_true_labels, outcome3_pred_labels)
    outcome3_f1 = f1_score(outcome3_true_labels, outcome3_pred_labels)

    result_dict = {}
    result_dict['mae'] = MAE_val
    result_dict['rmse'] = RMSE_val
    result_dict['outcome2_auc'] = outcome2_auc
    result_dict['outcome3_auc'] = outcome3_auc
    result_dict['rec loss'] = diagnostic_plot_reconstruction_loss_vals[-1]
    result_dict['rec loss plot vals'] = diagnostic_plot_reconstruction_loss_vals
    result_dict['pts_x_rank'] = model.factor0
    result_dict['temporal_phenotypes'] = model.factor1
    result_dict['static_phenotypes'] = model.Mfactor1
    result_dict['temporal_x_rank'] = model.factor2
    result_dict['weights'] = model.weights
    result_dict['patient bias'] = patient_bias
    result_dict['feature bias'] = feature_bias

    result_dict['outcome2_f1'] = outcome2_f1
    result_dict['outcome2_recall'] = outcome2_recall
    result_dict['outcome2_precision'] = outcome2_precision
    result_dict['outcome3_f1'] = outcome3_f1
    result_dict['outcome3_recall'] = outcome3_recall
    result_dict['outcome3_precision'] = outcome3_precision

    hyperparameters_dict = {}
    hyperparameters_dict['rank'] = model.rank
    hyperparameters_dict['n_epoch'] = model.n_epoch
    hyperparameters_dict['Lambda'] = model.Lambda
    hyperparameters_dict['lr'] = model.lr
    hyperparameters_dict['penalty_l1'] = model.penalty_l1
    hyperparameters_dict['penalty_l2'] = model.penalty_l2
    hyperparameters_dict['theta'] = model.theta

    return result_dict, hyperparameters_dict