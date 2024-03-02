import torch
from torch import nn
from timm import create_model
import torch.nn.functional as F
from .positional_encoding import PositionalEncoding2D


class ScouterAttention(nn.Module):
    def __init__(self, dim, num_concept, iters=3, eps=1e-8, vis=False, power=1, to_k_layer=3):
        super().__init__()
        self.num_slots = num_concept
        self.iters = iters
        self.eps = eps
        self.scale = dim ** (-0.5)

        # random seed init
        slots_mu = nn.Parameter(torch.randn(1, 1, dim))
        slots_sigma = nn.Parameter(torch.abs(torch.randn(1, 1, dim)))
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

    def forward(self, inputs_pe, inputs):
        b, n, d = inputs_pe.shape
        slots = self.initial_slots.expand(b, -1, -1)
        k, v = self.to_k(inputs_pe), inputs_pe
        print('slot.k.shape:', k.shape)
        for _ in range(self.iters):
            q = slots
            print('slot.q.shape:', q.shape)

            dots = torch.einsum('bid,bjd->bij', q, k) * self.scale
            dots = torch.div(dots, dots.sum(2).expand_as(dots.permute([2, 0, 1])).permute([1, 2, 0])) * \
                   dots.sum(2).sum(1).expand_as(dots.permute([1, 2, 0])).permute([2, 0, 1])
            attn = torch.sigmoid(dots)
            print('slot.dots.shape:', dots.shape)
            print('slot.attn.shape:', attn.shape)


            # print(torch.max(attn))
            # dsfds()

            attn2 = attn / (attn.sum(dim=-1, keepdim=True) + self.eps)
            print('slot.attn2.shape:', attn2.shape)
            updates = torch.einsum('bjd,bij->bid', inputs, attn2)
            print('slot.updates.shape:', updates.shape)
            break
        return updates, attn


class MainModel(nn.Module):
    def __init__(self, args, vis=False):
        super(MainModel, self).__init__()
        self.args = args
        self.pre_train = args.pre_train
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

        if not self.pre_train:
            self.conv1x1 = nn.Conv2d(self.num_features, hidden_dim, kernel_size=(1, 1), stride=(1, 1))
            self.norm = nn.BatchNorm2d(hidden_dim)
            self.position_emb = PositionalEncoding2D(hidden_dim)
            self.slots = ScouterAttention(hidden_dim, num_concepts, vis=self.vis)
            self.scale = 1
            self.cls = torch.nn.Linear(num_concepts, num_classes)
        else:
            self.fc = nn.Linear(self.num_features, num_classes)
            self.drop_rate = 0

    def forward(self, x):
        x = self.back_bone.forward_features(x)
        features = x
        # x = x.view(x.size(0), self.num_features, self.feature_size, self.feature_size)

        if not self.pre_train:
            x = self.conv1x1(x)
            x = self.norm(x)
            x = torch.relu(x)
            pe = self.position_emb(x)
            x_pe = x + pe

            b, c, h ,w = x.shape
            x = x.reshape((b, c, -1)).permute((0, 2, 1)) # shape: b, c, h*w
            x_pe = x_pe.reshape((b, c, -1)).permute((0, 2, 1)) # shape: b, c, h*w
            print('x.shape:', x.shape, 'x_pe.shape:', x_pe.shape)

            updates, attn = self.slots(x_pe, x)
            if self.args.cpt_activation == "att":
                cpt_activation = attn
            else:
                cpt_activation = updates
            attn_cls = self.scale * torch.sum(cpt_activation, dim=-1)
            cpt = self.activation(attn_cls)
            cls = self.cls(cpt)
            return (cpt - 0.5) * 2, cls, attn, updates
        else:
            x = F.adaptive_max_pool2d(x, 1).squeeze(-1).squeeze(-1)
            if self.drop_rate > 0:
                x = F.dropout(x, p=self.drop_rate, training=self.training)
            x = self.fc(x)
            return x, features