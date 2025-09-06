# Copyright (c) 2021 Rui Shu
import argparse
import numpy as np
import torch
import torch.utils.data
from codebase import utils as ut
from codebase.models import nns
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image

class SSVAE(nn.Module):
    def __init__(self, nn='v1', name='ssvae', gen_weight=1, class_weight=100):
        super().__init__()
        self.name = name
        self.z_dim = 64
        self.y_dim = 10
        self.gen_weight = gen_weight
        self.class_weight = class_weight
        nn = getattr(nns, nn)
        self.enc = nn.Encoder(self.z_dim, self.y_dim)
        self.dec = nn.Decoder(self.z_dim, self.y_dim)
        self.cls = nn.Classifier(self.y_dim)

        # Establecer prior como parámetro fijo adjunto al Módulo
        self.z_prior_m = torch.nn.Parameter(torch.zeros(1), requires_grad=False)
        self.z_prior_v = torch.nn.Parameter(torch.ones(1), requires_grad=False)
        self.z_prior = (self.z_prior_m, self.z_prior_v)

    def negative_elbo_bound(self, x):
        """
        Calcula el Límite Inferior de Evidencia (Evidence Lower Bound), KL y costos de Reconstrucción

        Args:
            x: tensor: (batch, dim): Observaciones

        Returns:
            nelbo: tensor: (): Límite inferior de evidencia negativo
            kl: tensor: (): Divergencia KL del ELBO al prior
            rec: tensor: (): Término de reconstrucción del ELBO
        """
#
        y_logits = self.cls(x)
        y_logprob = F.log_softmax(y_logits, dim=1) # log(q_\phi(y|x))
        y_prob = torch.softmax(y_logprob, dim=1) # q_\phi(y|x)
        # Duplicar y basado en el tamaño de batch de x. Luego duplicar x
        # Esto enumera todas las combinaciones posibles de x con etiquetas (0, 1, ..., 9)
        y = np.repeat(np.arange(self.y_dim), x.size(0))
        y = x.new(np.eye(self.y_dim)[y])
        x = ut.duplicate(x, self.y_dim)

        # -- Calculamos los 3 componentes de la perdida -- 
        
        #  1. Cálculo de pérdida del clasficador KL_y
        log_prior_y = torch.full_like(y_logprob, np.log(1.0 / self.y_dim))
        kl_y_total = ut.kl_cat(y_prob, y_logprob, log_prior_y).sum()

        #  2. Calculamos la pérdida latente (KL_z)  (Cálculo preliminar por par)
        #Se calculan sus valores preliminares para cada par (x,y) dell lote expandido.
        m, v = self.enc(x, y)
        kl_z_pares = ut.kl_normal(m, v, torch.zeros_like(m), torch.ones_like(v))

        #  3. Pérdida de reconstrucción (Rec)   (Cálculo preliminar por par)       
        z = ut.sample_gaussian(m, v) #Muestreo de Monte Carlo para z, un 'z' para cada par (x,y).
        logits_hat_x = self.dec(z, y)
        rec_pares = -ut.log_bernoulli_with_logits(x, logits_hat_x)

        # Suma Ponderada y finalizamos el cálculo de KL_z y Rec
        # Usamos q(y|x) para ponderar las péridas preliminares y obtener los totales de Kl_x y Rec
        batch_size = y_prob.size(0)
        kl_z_pares = kl_z_pares.view(self.y_dim, batch_size)
        rec_pares = rec_pares.view(self.y_dim, batch_size)
        kl_z_total = (y_prob.t() * kl_z_pares).sum()
        rec_total = (y_prob.t() * rec_pares).sum()


        #  Se promedian las pérdidas totales sobre el tamaño del lote original
        kl_y = kl_y_total / batch_size
        kl_z = kl_z_total / batch_size
        rec = rec_total / batch_size

    	#Pérdida total
        nelbo = kl_y + kl_z + rec


        return nelbo, kl_z, kl_y, rec

    def classification_cross_entropy(self, x, y):
        y_logits = self.cls(x)
        return F.cross_entropy(y_logits, y.argmax(1))

    def loss(self, x, xl, yl):
        if self.gen_weight > 0:
            nelbo, kl_z, kl_y, rec = self.negative_elbo_bound(x)
        else:
            nelbo, kl_z, kl_y, rec = [0] * 4
        ce = self.classification_cross_entropy(xl, yl)
        loss = self.gen_weight * nelbo + self.class_weight * ce

        summaries = dict((
            ('train/loss', loss),
            ('class/ce', ce),
            ('gen/elbo', -nelbo),
            ('gen/kl_z', kl_z),
            ('gen/kl_y', kl_y),
            ('gen/rec', rec),
        ))

        return loss, summaries

    def compute_sigmoid_given(self, z, y):
        logits = self.dec(z, y)
        return torch.sigmoid(logits)

    def sample_z(self, batch):
        return ut.sample_gaussian(self.z_prior[0].expand(batch, self.z_dim),
                                  self.z_prior[1].expand(batch, self.z_dim))

    def sample_x_given(self, z, y):
        return torch.bernoulli(self.compute_sigmoid_given(z, y))
