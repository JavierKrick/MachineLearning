# Copyright (c) 2021 Rui Shu

import torch
from codebase import utils as ut
from codebase.models import nns
from torch import nn
from torch.nn import functional as F

torch.cuda.empty_cache()

class VAE(nn.Module):
    def __init__(self, nn='v1', name='vae', z_dim=2):
        super().__init__()
        self.name = name
        self.z_dim = z_dim
        nn = getattr(nns, nn)
        self.enc = nn.Encoder(self.z_dim)
        self.dec = nn.Decoder(self.z_dim)

        # Fija el prior como un parámetro fijo adjunto al Módulo
        self.z_prior_m = torch.nn.Parameter(torch.zeros(1), requires_grad=False)
        self.z_prior_v = torch.nn.Parameter(torch.ones(1), requires_grad=False)
        self.z_prior = (self.z_prior_m, self.z_prior_v)

    def negative_elbo_bound(self, x):
        """
        Calcula la Evidencia Inferior Acotada (ELBO), el KL y los costos de Reconstrucción

        Args:
            x: tensor: (batch, dim): Observaciones

        Returns:
            nelbo: tensor: (): Evidencia inferior acotada negativa
            kl: tensor: (): Divergencia KL de ELBO al prior
            rec: tensor: (): Término de reconstrucción de ELBO
        """
        ################################################################################
        # TODO: Modifique/completar el código aquí
        # Calcule la Evidencia Inferior Acotada negativa y su descomposición en KL y Rec
        #
        # Notar que nelbo = kl + rec
        #
        # Todas las salidas deben ser escalares
        ################################################################################
        from codebase.utils import sample_gaussian, kl_normal

        m, v = self.enc(x)  # -> media y varianza
        # Muestreo Monte Carlo
        z = sample_gaussian(m, v)

        # Calcular la divergencia KL con el prior
        kl_divergence = kl_normal(m, v, self.z_prior_m, self.z_prior_v).sum()

        x_logits = self.dec(z) #Pasar z por el decoder

        reconstruction_loss = F.binary_cross_entropy_with_logits(x_logits, x, reduction='sum')

        # Promediar sobre el tamaño del batch
        batch_size = x.size(0)
        rec = reconstruction_loss / batch_size
        
        # KL term
        kl = kl_divergence / batch_size

        # ELBO negativa promedio
        nelbo = rec + kl


        ################################################################################
        # Fin de la modificación del código
        ################################################################################
        return nelbo, kl, rec

    def negative_iwae_bound(self, x, iw):
        """
        Calcula la cota del Autoencoder de Importancia Ponderada (IWAE)
        Adicionalmente, también calcula los términos KL y reconstrucción de la ELBO

        Args:
            x: tensor: (batch, dim): Observaciones
            iw: int: (): Número de muestras de importancia

        Returns:
            niwae: tensor: (): Cota negativa de IWAE
            kl: tensor: (): Divergencia KL de ELBO al prior
            rec: tensor: (): Término de reconstrucción de ELBO
        """
        m, v = self.enc(x)
        m = ut.duplicate(m, iw)
        v = ut.duplicate(v, iw)
        x = ut.duplicate(x, iw)
        z = ut.sample_gaussian(m, v)
        logits = self.dec(z)
        kl = ut.log_normal(z, m, v) - ut.log_normal(z, self.z_prior[0], self.z_prior[1])
        rec = -ut.log_bernoulli_with_logits(x=x, logits=logits)
        nelbo = kl + rec
        iwae = ut.log_mean_exp(-nelbo.reshape(iw, -1), dim=0)
        niwae = -iwae
        niwae = torch.mean(niwae)
        kl = torch.mean(kl)
        rec = torch.mean(rec)
        return niwae, kl, rec

    def loss(self, x):
        nelbo, kl, rec = self.negative_elbo_bound(x)
        loss = nelbo

        summaries = dict((
            ('train/loss', nelbo),
            ('gen/elbo', -nelbo),
            ('gen/kl_z', kl),
            ('gen/rec', rec),
        ))

        return loss, summaries

    def sample_sigmoid(self, batch):
        z = self.sample_z(batch)
        return self.compute_sigmoid_given(z)

    def compute_sigmoid_given(self, z):
        logits = self.dec(z)
        return torch.sigmoid(logits)

    def sample_z(self, batch):
        return ut.sample_gaussian(
            self.z_prior[0].expand(batch, self.z_dim),
            self.z_prior[1].expand(batch, self.z_dim))

    def sample_x(self, batch):
        z = self.sample_z(batch)
        return self.sample_x_given(z)

    def sample_x_given(self, z):
        return torch.bernoulli(self.compute_sigmoid_given(z))
