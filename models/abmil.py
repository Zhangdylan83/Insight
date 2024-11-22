import torch
import torch.nn as nn
import torch.nn.functional as F

class Attention(nn.Module):
    def __init__(self, feature_dim):
        super(Attention, self).__init__()
        self.M = feature_dim  # Feature dimension
        self.L = 128
        self.ATTENTION_BRANCHES = 1

        self.attention = nn.Sequential(
            nn.Linear(self.M, self.L),  # matrix V
            nn.Tanh(),
            nn.Linear(self.L, self.ATTENTION_BRANCHES)  # matrix w (or vector w if self.ATTENTION_BRANCHES==1)
        )

        self.classifier = nn.Sequential(
            nn.Linear(self.M * self.ATTENTION_BRANCHES, 1),
            nn.Sigmoid()
        )

    def forward(self, H):
        A = self.attention(H)  # KxATTENTION_BRANCHES  H expected [num_instances, feature_emb] from chatgpt
        A = torch.transpose(A, 1, 0)  # ATTENTION_BRANCHESxK
        A = F.softmax(A, dim=1)  # softmax over K

        Z = torch.mm(A, H)  # ATTENTION_BRANCHESxM

        Y_prob = self.classifier(Z)
        Y_hat = torch.ge(Y_prob, 0.5).float()

        results_dict = {
            "Y_prob": Y_prob, 
            "Y_hat": Y_hat,
            "attention_weights": A
        }

        return results_dict

    # AUXILIARY METHODS
    def calculate_classification_error(self, H, Y):
        Y = Y.float()
        _, Y_hat, _ = self.forward(H)
        error = 1. - Y_hat.eq(Y).cpu().float().mean().item()

        return error, Y_hat

    def calculate_objective(self, H, Y):
        Y = Y.float()
        Y_prob, _, A = self.forward(H)
        Y_prob = torch.clamp(Y_prob, min=1e-5, max=1. - 1e-5)
        neg_log_likelihood = -1. * (Y * torch.log(Y_prob) + (1. - Y) * torch.log(1. - Y_prob))  # negative log bernoulli

        return neg_log_likelihood, A

class GatedAttention(nn.Module):
    def __init__(self, feature_dim):
        super(GatedAttention, self).__init__()
        self.M = feature_dim  # Feature dimension
        self.L = 128
        self.ATTENTION_BRANCHES = 1

        self.attention_V = nn.Sequential(
            nn.Linear(self.M, self.L),  # matrix V
            nn.Tanh()
        )

        self.attention_U = nn.Sequential(
            nn.Linear(self.M, self.L),  # matrix U
            nn.Sigmoid()
        )

        self.attention_w = nn.Linear(self.L, self.ATTENTION_BRANCHES)  # matrix w (or vector w if self.ATTENTION_BRANCHES==1)

        self.classifier = nn.Sequential(
            nn.Linear(self.M * self.ATTENTION_BRANCHES, 1),
            nn.Sigmoid()
        )

    def forward(self, H):
        A_V = self.attention_V(H)  # KxL
        #print(H.shape, "H")
        #print(A_V.shape, "A_V")
        A_U = self.attention_U(H)  # KxL
        A = self.attention_w(A_V * A_U)  # element wise multiplication # KxATTENTION_BRANCHES
        A = torch.transpose(A, 1, 0)  # ATTENTION_BRANCHESxK
        A = F.softmax(A, dim=1)  # softmax over K
        #print("Shape of A:", A.shape)  #Shape of A: torch.Size([1, K])
        
        #print("Shape of H:", H.shape)  # Shape of H: torch.Size([K, 1024])
        

        Z = torch.mm(A, H)  # ATTENTION_BRANCHESxM 
        #print(Z.shape, "Z") torch.Size([1, 1024])
        Y_prob = self.classifier(Z)
        #print(Y_prob.shape, "Y_prob")
        Y_hat = torch.ge(Y_prob, 0.5).float()

        results_dict = {
            "Y_prob": Y_prob, 
            "Y_hat": Y_hat,
            "attention_weights": A
        }

        return results_dict

    # AUXILIARY METHODS
    def calculate_classification_error(self, H, Y):
        Y = Y.float()
        _, Y_hat, _ = self.forward(H)
        error = 1. - Y_hat.eq(Y).cpu().float().mean().item()

        return error, Y_hat
    '''
    def calculate_objective(self, H, Y):
        Y = Y.float()
        Y_prob, _, A = self.forward(H)
        Y_prob = torch.clamp(Y_prob, min=1e-5, max=1. - 1e-5)
        neg_log_likelihood = -1. * (Y * torch.log(Y_prob) + (1. - Y) * torch.log(1. - Y_prob))  # negative log bernoulli

        return neg_log_likelihood, A
    '''
    def calculate_objective(self, Y_prob, Y):
        Y = Y.float()
        Y_prob = torch.clamp(Y_prob, min=1e-5, max=1. - 1e-5)
        neg_log_likelihood = -1. * (Y * torch.log(Y_prob) + (1. - Y) * torch.log(1. - Y_prob))  # negative log bernoulli

        return neg_log_likelihood
