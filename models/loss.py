import torch
import torch.nn.functional as F
from torch.autograd import Variable


def contrastive_loss(preds, labels, use_similarity=False):
    # mask shape: [b, b]
    # mask[i, j] == True if item i and item j has the same class label, False otherwise
    mask = Variable(labels @ labels.T).bool()

    if use_similarity:
        preds = preds / preds.norm(dim=-1)[:, None]
        dot = preds @ preds.T
        loss = - torch.log(dot * mask + (1 - dot) * ~mask)
    else:
        dot = preds @ preds.T
        exp_dot = torch.exp(dot)
        loss = (torch.log(1 + exp_dot) - mask * dot)

    # Add weight according to Supp 2.3
    loss[mask] = loss[mask] * (torch.sum(mask) / mask.numel())
    loss[~mask] = loss[~mask] * (torch.sum(~mask) / mask.numel())

    loss = torch.mean(loss)
    return loss


def consistency_loss(features, attention):
    '''For each concept, the attention should produce similar aggregated features'''
    b, num_concepts, spatial = attention.size()
    loss = 0
    for i in range(num_concepts):
        f_i = features[:, i, :]
        attention_i = attention[:, i, :].sum(-1)
        indices = attention_i > attention_i.mean()
        selected_f = f_i[indices]
        loss += F.cosine_similarity(selected_f[None, :, :], selected_f[:, None, :], dim=-1).mean()
    return - loss / num_concepts


def batch_cpt_discriminate(features, attention):
    b, num_concepts, d = features.shape
    mean_features = [] # list of len num_concepts, each item has shape [1, dim]
    for i in range(num_concepts):
        f_i = features[:, i, :] # shape: [b, dim]
        concept_logits = attention[:, i, :].sum(-1) # shape: [b, h*w]
        indices = concept_logits > concept_logits.mean()

        selected_f = f_i[indices]
        mean_features.append(torch.mean(selected_f, dim=0))
    
    mean_features = torch.stack(mean_features, dim=0) # shape: [num_concepts, dim]
    sim = F.cosine_similarity(mean_features[None, :, :], mean_features[:, None, :], dim=-1)
    return sim.mean()
