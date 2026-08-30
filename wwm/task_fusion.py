"""
Task-specific learnable fusion for multi-level features.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class TaskSpecificFusion(nn.Module):
    """
    Learnable gated fusion of multi-level encoder features for downstream tasks.

    Instead of uniform averaging, learns task-specific weights for each layer.
    Compression tasks may benefit from different layer combinations than beam/localization.
    """

    def __init__(self, num_layers=4, hidden_dim=768, fusion_mode='static'):
        """
        Args:
            num_layers: Number of encoder layers to fuse
            hidden_dim: Feature dimension
            fusion_mode: 'static' (learnable fixed weights) or 'dynamic' (content-based)
        """
        super().__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.fusion_mode = fusion_mode

        if fusion_mode == 'static':
            # Learnable fixed weights (one scalar per layer)
            self.gates = nn.Parameter(torch.ones(num_layers))
        elif fusion_mode == 'dynamic':
            # Content-based dynamic weights
            self.gate_net = nn.Sequential(
                nn.Linear(hidden_dim, num_layers * 2),
                nn.ReLU(),
                nn.Linear(num_layers * 2, num_layers)
            )
        else:
            raise ValueError(f"Unknown fusion_mode: {fusion_mode}")

    def forward(self, multilevel_features):
        """
        Args:
            multilevel_features: List[Tensor] of shape [B, T, D], length = num_layers

        Returns:
            fused: Tensor of shape [B, T, D]
        """
        if not isinstance(multilevel_features, (list, tuple)):
            # Single level, no fusion needed
            return multilevel_features

        if len(multilevel_features) != self.num_layers:
            raise ValueError(
                f"Expected {self.num_layers} levels, got {len(multilevel_features)}"
            )

        # Stack features: [num_layers, B, T, D]
        stacked = torch.stack(multilevel_features, dim=0)

        if self.fusion_mode == 'static':
            # Static learnable weights
            weights = F.softmax(self.gates, dim=0)  # [num_layers]
            weights = weights.view(-1, 1, 1, 1)  # [num_layers, 1, 1, 1]
            fused = (stacked * weights).sum(dim=0)  # [B, T, D]

        elif self.fusion_mode == 'dynamic':
            # Dynamic weights based on content
            # Use mean-pooled features as context
            context = stacked.mean(dim=2)  # [num_layers, B, D]
            context = context.mean(dim=0)  # [B, D]
            weights = self.gate_net(context)  # [B, num_layers]
            weights = F.softmax(weights, dim=-1)  # [B, num_layers]

            # Apply weights: [B, num_layers, 1, 1] * [num_layers, B, T, D]
            weights = weights.transpose(0, 1).unsqueeze(-1).unsqueeze(-1)
            fused = (stacked * weights).sum(dim=0)  # [B, T, D]

        return fused

    def get_fusion_weights(self):
        """Return current fusion weights for analysis."""
        if self.fusion_mode == 'static':
            return F.softmax(self.gates, dim=0).detach().cpu().numpy()
        else:
            return None  # Dynamic weights vary per sample


class SupervisedBottleneck(nn.Module):
    """
    Supervised learnable 1D bottleneck for compression tasks.

    Instead of fixed PCA projection, learns the optimal projection direction
    end-to-end with the reconstruction objective.
    """

    def __init__(self, latent_dim=768, bottleneck_dim=1,
                 use_pca_init=True, l2_reg=0.0):
        """
        Args:
            latent_dim: Input feature dimension
            bottleneck_dim: Compressed dimension (typically 1 for extreme compression)
            use_pca_init: Whether to initialize with PCA (done externally)
            l2_reg: L2 regularization on projection weights
        """
        super().__init__()
        self.latent_dim = latent_dim
        self.bottleneck_dim = bottleneck_dim
        self.l2_reg = l2_reg

        # Learnable projection (can be initialized with PCA)
        self.reduce = nn.Linear(latent_dim, bottleneck_dim, bias=True)
        self.expand = nn.Linear(bottleneck_dim, latent_dim, bias=True)

    def forward(self, x):
        """
        Args:
            x: [B, T, D] input features

        Returns:
            x_rec: [B, T, D] reconstructed features
            z: [B, T, bottleneck_dim] compressed features
        """
        z = self.reduce(x)  # [B, T, bottleneck_dim]
        x_rec = self.expand(z)  # [B, T, D]
        return x_rec, z

    def projection_regularization(self):
        """L2 regularization on projection weights to prevent collapse."""
        if self.l2_reg > 0:
            return self.l2_reg * (
                self.reduce.weight.pow(2).sum() +
                self.expand.weight.pow(2).sum()
            )
        return 0.0

    def initialize_from_pca(self, data_samples, device):
        """
        Initialize projection with PCA on data samples.

        Args:
            data_samples: [N, D] tensor of features
            device: torch device
        """
        from sklearn.decomposition import PCA

        # Fit PCA
        pca = PCA(n_components=self.bottleneck_dim)
        pca.fit(data_samples.cpu().numpy())

        # Extract components
        components = torch.from_numpy(pca.components_).float()  # [bottleneck_dim, D]
        mean = torch.from_numpy(pca.mean_).float()  # [D]

        # Compute scaling
        explained_var = torch.from_numpy(pca.explained_variance_).float()  # [bottleneck_dim]
        std = explained_var.clamp_min(1e-6).sqrt()

        # Set weights
        with torch.no_grad():
            reduce_weight = components / std[:, None]  # [bottleneck_dim, D]
            reduce_bias = -(reduce_weight @ mean)  # [bottleneck_dim]

            expand_weight = components.T * std[None, :]  # [D, bottleneck_dim]
            expand_bias = mean  # [D]

            self.reduce.weight.copy_(reduce_weight.to(device))
            self.reduce.bias.copy_(reduce_bias.to(device))
            self.expand.weight.copy_(expand_weight.to(device))
            self.expand.bias.copy_(expand_bias.to(device))

        print(f"Initialized {self.bottleneck_dim}D bottleneck with PCA")
        print(f"  Explained variance: {explained_var.tolist()}")


class InformationBottleneck1D(nn.Module):
    """
    Variational Information Bottleneck for 1D compression.

    Uses variational inference to learn a stochastic 1D projection
    that balances reconstruction quality and information compression.
    """

    def __init__(self, latent_dim=768, beta=1e-3):
        """
        Args:
            latent_dim: Input feature dimension
            beta: KL divergence weight (information bottleneck strength)
        """
        super().__init__()
        self.latent_dim = latent_dim
        self.beta = beta

        # Encode to 1D mean and log-variance
        self.encoder = nn.Linear(latent_dim, 2)

        # Decode from 1D
        self.decoder = nn.Linear(1, latent_dim)

    def forward(self, x, return_kl=False):
        """
        Args:
            x: [B, T, D] input features
            return_kl: Whether to return KL divergence

        Returns:
            x_rec: [B, T, D] reconstructed features
            z: [B, T, 1] sampled 1D codes
            kl_loss: scalar KL divergence (if return_kl=True)
        """
        # Encode to mean and log-variance
        stats = self.encoder(x)  # [B, T, 2]
        mean, logvar = stats.chunk(2, dim=-1)  # Each [B, T, 1]

        # Reparameterization trick
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(mean)
            z = mean + eps * std
        else:
            z = mean  # Use mean at test time

        # Decode
        x_rec = self.decoder(z)  # [B, T, D]

        if return_kl:
            # KL divergence from standard normal
            kl_loss = -0.5 * torch.sum(
                1 + logvar - mean.pow(2) - logvar.exp(),
                dim=(1, 2)  # Sum over T and D
            ).mean()  # Average over batch
            return x_rec, z, kl_loss
        else:
            return x_rec, z

    def total_loss(self, x_rec, z, kl_loss, csi_pred, csi_target,
                   compression_loss_fn, model_args):
        """
        Combined reconstruction + KL loss.

        Args:
            x_rec: [B, T, D] reconstructed features
            z: [B, T, 1] compressed codes (unused here)
            kl_loss: scalar KL divergence
            csi_pred: [B, future, H, W] predicted CSI
            csi_target: [B, future, H, W] target CSI
            compression_loss_fn: Function to compute CSI reconstruction loss
            model_args: Model arguments

        Returns:
            total_loss: scalar
            loss_dict: Dictionary of loss components
        """
        # CSI reconstruction loss
        recon_loss = compression_loss_fn(csi_pred, csi_target, model_args)

        # Total loss
        total = recon_loss + self.beta * kl_loss

        return total, {
            'recon_loss': recon_loss.item(),
            'kl_loss': kl_loss.item(),
            'total_loss': total.item()
        }
