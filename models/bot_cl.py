import torch
import numpy as np
from torch import nn
from PIL import Image
from timm import create_model
import torch.nn.functional as F

from .positional_encoding import PositionalEncoding2D


class ScouterAttention(nn.Module):
    def __init__(self, args, dim, num_concept, iters=3, eps=1e-8, vis=False, power=1, to_k_layer=3):
        super().__init__()
        self.args = args
        self.num_slots = num_concept
        self.eps = eps
        self.scale = dim ** -0.5

        # random seed init
        slots_mu = torch.randn(1, 1, dim)
        slots_sigma = torch.abs(torch.randn(1, 1, dim))
        mu = slots_mu.expand(1, self.num_slots, -1)
        sigma = slots_sigma.expand(1, self.num_slots, -1)
        self.initial_slots = nn.Parameter(torch.normal(mu, sigma))

        # K layer init
        to_k_layer_list = [nn.Linear(dim, dim)]
        for to_k_layer_id in range(1, to_k_layer):
            to_k_layer_list.append(nn.ReLU(inplace=True))
            to_k_layer_list.append(nn.Linear(dim, dim))
        self.to_k = nn.Sequential(
            *to_k_layer_list
        )

        self.vis = vis
        self.power = power

    def forward(self, inputs_pe, inputs, weight=None, things=None):
        b, n, d = inputs_pe.shape
        slots = self.initial_slots.expand(b, -1, -1)
        k, v = self.to_k(inputs_pe), inputs_pe
        for _ in range(self.iters):
            q = slots

            dots = torch.einsum('bid,bjd->bij', q, k) * self.scale
            scale_fct = dots.sum(2).sum(1)[:, None, None].expand_as(dots)
            dots = torch.div(dots, dots.sum(2).unsqueeze(-1).expand_as(dots)) * scale_fct
            attn = torch.sigmoid(dots)

            attn2 = attn / (attn.sum(dim=-1, keepdim=True) + self.eps)
            updates = torch.einsum('bjd,bij->bid', inputs, attn2)
        if self.vis:
            slots_vis_raw = attn.clone()
            vis(slots_vis_raw, "vis", self.args.feature_size, weight, things)
        return updates, attn


def vis(slots_vis_raw, loc, size, weight=None, things=None):
    b = slots_vis_raw.size()[0]
    for i in range(b):
        slots_vis = slots_vis_raw[i]
        if weight is not None:
            Nos = weight[1]
            weight = weight[0]
            slots_vis = slots_vis * weight.unsqueeze(-1)

            overall = slots_vis.sum(0)
            overall = (((overall - overall.min()) / (overall.max() - overall.min())) * 255.).reshape((int(size), int(size)))
            overall = (overall.cpu().detach().numpy()).astype(np.uint8)
            overall = Image.fromarray(overall, mode='L').resize([224, 224], resample=Image.BILINEAR)
            overall.save(f'{loc}/overall_{Nos}.png')

        else:
            slots_vis = ((slots_vis - slots_vis.min()) / (slots_vis.max() - slots_vis.min()) * 255.).reshape(
                slots_vis.shape[:1] + (int(size), int(size)))

            slots_vis = (slots_vis.cpu().detach().numpy()).astype(np.uint8)
            for id, image in enumerate(slots_vis):

                image = Image.fromarray(image, mode='L').resize([224, 224], resample=Image.BILINEAR)
                if things is not None:
                    order, category, cpt_num = things
                    loc2 = f"vis_pp/cpt{cpt_num}/"
                    if id == cpt_num:
                        image.save(loc2 + f'mask_{order}_{category}.png')
                        break
                    else:
                        continue
                image.save(f'{loc}/{i}_slot_{id:d}.png')


class ResNetClassifier(nn.Module):
    def __init__(self, model_name: str='resnet18', dim=512, num_classes=200):
        super(ResNetClassifier, self).__init__()
        self.resnet = create_model(model_name, pretrained=True)
        self.fc = nn.Linear(dim, num_classes)
        self.dropout_rate = 0

    def forward(self, x):
        x = self.resnet.forward_features(x)
        x = F.adaptive_max_pool2d(x, 1).squeeze(-1).squeeze(-1)
        if self.dropout_rate > 0:
            x = F.dropout(x, p=self.drop_rate, training=self.training)
        x = self.fc(x)
        return x


class BotCl(nn.Module):
    def __init__(self, args, vis=False):
        super(BotCl, self).__init__()
        self.args = args
        if "18" not in args.base_model:
            self.num_features = 2048
        else:
            self.num_features = 512
        self.feature_size = args.feature_size
        self.drop_rate = 0.0
        hidden_dim = 128
        num_concepts = args.num_cpt
        num_classes = args.num_classes
        self.back_bone = create_model('resnet18', pretrained=True)
        self.activation = nn.Tanh()
        self.vis = vis

        self.scale = 1
        self.conv1x1 = nn.Conv2d(self.num_features, hidden_dim,
                                 kernel_size=(1, 1), stride=(1, 1))
        self.norm = nn.BatchNorm2d(hidden_dim)
        self.position_emb = PositionalEncoding2D(hidden_dim)
        self.slot_attention = ScouterAttention(hidden_dim, num_concepts, vis=self.vis)
        self.classifier = torch.nn.Linear(num_concepts, num_classes)


    def forward(self, x):
        x = self.back_bone.forward_features(x)

        x = self.conv1x1(x)
        x = self.norm(x)
        x = torch.relu(x)
        pe = self.position_emb(x)
        x_pe = x + pe

        b, c = x.shape[:2]
        x = x.reshape((b, c, -1)).permute((0, 2, 1)) # shape: [b, c, h*w]
        x_pe = x_pe.reshape((b, c, -1)).permute((0, 2, 1)) # shape: [b, c, h*w]

        # concepts shape: [b, num_concepts, dim];
        # attention_scores shape: [b, num_concepts, num_patches]
        concepts, attention_scores = self.slot_attention(x_pe, x)
        if self.args.cpt_activation == "att":
            cpt_activation = attention_scores
        else:
            cpt_activation = concepts
        attn_cls = self.scale * torch.sum(cpt_activation, dim=-1)

        concept_logits = self.activation(attn_cls)
        class_logits = self.classifier(concept_logits)
        concept_logits_hat = (concept_logits - 0.5) * 2

        return concept_logits_hat, class_logits, attention_scores, concepts
