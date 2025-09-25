import torch
from torch.distributions import MultivariateNormal    
import gpytorch

class GPWithMissingSampler(object):
    def __init__(self, kernel):
        self.kernel = kernel
        self.feature_groups = None

    def _generate_missingness(self, x, num_observed=None, seed=None, missing_rate=None):
        """Generate random missingness mask for each sample in batch.
        """
        batch_size, seq_len, num_features = x.shape
        
        if num_observed is None:
            # Randomly mask features
            if seed is not None:
                generator = torch.Generator().manual_seed(seed)
                unif = torch.rand(batch_size, seq_len, num_features, generator=generator)
                if missing_rate is not None:
                    ref = torch.full((batch_size, seq_len, 1), missing_rate)
                else:
                    ref = torch.rand(batch_size, seq_len, 1, generator=generator)
            else:
                unif = torch.rand(batch_size, seq_len, num_features)
                if missing_rate is not None:
                    ref = torch.full((batch_size, seq_len, 1), missing_rate)
                else:   
                    ref = torch.rand(batch_size, seq_len, 1)
                    
            mask = (unif > ref).float().to(x.device)
        else:
            if num_observed == 0:
                mask = torch.zeros_like(x)
            elif num_observed == num_features:
                mask = torch.ones_like(x)
            else:
                if seed is not None:
                    generator = torch.Generator(device=x.device).manual_seed(seed)
                    rand_scores = torch.rand(batch_size, seq_len, num_features, device=x.device, generator=generator)
                else:
                    rand_scores = torch.rand(batch_size, seq_len, num_features, device=x.device)
                
                _, observed_features = torch.topk(rand_scores, num_observed, dim=2)
                mask = torch.zeros_like(x)
                mask.scatter_(2, observed_features, 1.0)
                
        return mask

    def sample(self,
            batch_size=16,
            num_ctx=None,
            num_tar=None,
            max_num_points=500,
            x_range=(-2, 2),
            x_dim=1,
            device='cpu',
            num_observed=None,
            seed=None,
            max_p_missing=0.5,
            mode='train'):

        batch = {}
        num_ctx = num_ctx or 50
        if num_tar is None:
            num_tar = max_num_points - num_ctx

        num_points = num_ctx + num_tar  # N = Nc + Nt

        if seed is not None:
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)

            x = x_range[0] + (x_range[1] - x_range[0]) \
                * torch.rand([batch_size, num_points, x_dim], device=device)

            if mode == 'train':
                r = torch.ones_like(x)
                missing_probs = torch.rand(batch_size, 1, x_dim, device=device) * max_p_missing  # [B,1,Dx]
                rand_vals = torch.rand([batch_size, num_points, x_dim], device=device)
                r = (rand_vals > missing_probs).float()
            elif mode == 'test':
                r = torch.ones_like(x)
                missing_probs = torch.rand(batch_size, 1, x_dim, device=device) * max_p_missing  # [B,1,Dx]
                rand_vals = torch.rand([batch_size, num_ctx, x_dim], device=device)
                r[:, :num_ctx] = (rand_vals > missing_probs).float()  # [B,Nc,Dx]
            else:
                raise ValueError(f"Invalid mode: {mode}")
            
            mask = torch.ones_like(x)
            mask[:, num_ctx:] = self._generate_missingness(x[:, num_ctx:], num_observed, seed=seed)

            cov = self.kernel(x)  # Use original x for kernel computation [B,N,N]
            mean = torch.zeros(batch_size, num_points, device=device)  # [B,N]
            y = MultivariateNormal(mean, scale_tril=cov).rsample().unsqueeze(-1)  # [B,N,Dy=1]

        else:
            # Generate features
            x = x_range[0] + (x_range[1] - x_range[0]) \
                    * torch.rand([batch_size, num_points, x_dim], device=device)  # [B,N,Dx]
            
            # Context mask: Each feature in each sample is missing retrospectively with random probability in [0, 0.5)
            if mode == 'train':
                missing_probs = torch.rand(batch_size, 1, x_dim, device=device) * max_p_missing  # [B,1,Dx]
                rand_vals = torch.rand([batch_size, num_points, x_dim], device=device)
                r = (rand_vals > missing_probs).float()
            elif mode == 'test':
                r = torch.ones_like(x)
                missing_probs = torch.rand(batch_size, 1, x_dim, device=device) * max_p_missing  # [B,1,Dx]
                rand_vals = torch.rand([batch_size, num_ctx, x_dim], device=device)
                r[:, :num_ctx] = (rand_vals > missing_probs).float()  # [B,Nc,Dx]
            else:
                raise ValueError(f"Invalid mode: {mode}")

        # Generate target missingness mask: 1 = observed, 0 = missing
            mask = torch.ones_like(x)
            mask[:, num_ctx:] = self._generate_missingness(x[:, num_ctx:], num_observed)

            cov = self.kernel(x)  # Use original x for kernel computation [B,N,N]
            mean = torch.zeros(batch_size, num_points, device=device)  # [B,N]
            y = MultivariateNormal(mean, scale_tril=cov).rsample().unsqueeze(-1)  # [B,N,Dy=1]

        batch['x'] = x
        batch['xc'] = x[:, :num_ctx]  # [B,Nc,2*Dx]
        batch['xt'] = x[:, num_ctx:]  # [B,Nt,2*Dx]

        # batch_size * num_points * num_points
        batch['y'] = y # [B,N,Dy=1]
        batch['yc'] = batch['y'][:, :num_ctx]  # [B,Nc,1]
        batch['yt'] = batch['y'][:, num_ctx:]  # [B,Nt,1]

        # Optionally, also return the mask for downstream use
        batch['mask'] = mask
        batch['maskc'] = mask[:, :num_ctx]
        batch['maskt'] = mask[:, num_ctx:]

        batch['r'] = r
        batch['rc'] = r[:, :num_ctx]
        batch['rt'] = r[:, num_ctx:]

        return batch
    
    def sample_context(self, batch, num_ctx):
        new_ctx = batch['xc'][:, :num_ctx].clone()
        new_ctx_y = batch['yc'][:, :num_ctx].clone()
        new_ctx_mask = batch['maskc'][:, :num_ctx].clone()
        new_ctx_r = batch['rc'][:, :num_ctx].clone()

        x = torch.cat([new_ctx, batch['xt']], dim=1)

        new_batch = {}
        new_batch['x'] = x
        new_batch['xc'] = new_ctx  # [B,Nc,2*Dx]
        new_batch['xt'] = batch['xt']

        new_batch['y'] = batch['y']
        new_batch['yc'] = new_ctx_y
        new_batch['yt'] = batch['yt']

        new_batch['mask'] = torch.cat([new_ctx_mask, batch['maskt']], dim=1)
        new_batch['maskc'] = new_ctx_mask
        new_batch['maskt'] = batch['maskt']

        new_batch['r'] = batch['r']
        new_batch['rc'] = new_ctx_r
        new_batch['rt'] = batch['rt']

        return new_batch
    
    def mask_features(self, batch, num_observed=None, mask=None, seed=None, missing_rate=None):
        x = batch['xt']

        # Sample target state missingness mask if not provided
        if mask is None:
            mask = self._generate_missingness(x, num_observed=num_observed, seed=seed, missing_rate=missing_rate)

        new_batch = {
            'x': batch['x'],
            'xc': batch['xc'],
            'xt': batch['xt'],
            'y': batch['y'],
            'yc': batch['yc'],
            'yt': batch['yt'],
            'mask': torch.cat((batch['maskc'], mask), dim=1),
            'maskc': batch['maskc'],
            'maskt': mask,
            'r': batch['r'],
            'rc': batch['rc'],
            'rt': batch['rt']
        }

        return new_batch
    
    def initialize_mask(self, batch):
        maskt = torch.zeros_like(batch['maskt'])
        return maskt
    
    def acquire_features(self, batch, mask, action=None, seed=None):
        batch_size, seq_len, x_dim = batch['maskt'].shape
        xt = batch['xt']
        rt = batch['rt'] if 'rt' in batch else None

        # If mask is not provided, initialize it to all zeros
        if mask is None:
            mask = torch.zeros_like(batch['maskt'])

        # If action is not provided, randomly select an action
        if action is None:
            # Create new mask
            new_mask = mask.clone()

            if rt is not None:
                available_mask = (mask == 0) & (rt == 1)  # shape: (B, T, F)
            else:
                available_mask = (mask == 0)

            if seed is not None:
                generator = torch.Generator(device=mask.device).manual_seed(seed)

            for b in range(batch_size):
                for t in range(seq_len):
                    available_feats = torch.where(available_mask[b, t])[0]
                    if len(available_feats) > 0:
                        if seed is not None:
                            idx = torch.randint(0, len(available_feats), (1,), device=mask.device, generator=generator)
                        else:
                            idx = torch.randint(0, len(available_feats), (1,), device=mask.device)
                        new_mask[b, t, available_feats[idx]] = 1.0

        else:
            new_mask = mask.clone()
            batch_idx = torch.arange(batch_size, device=action.device)[:, None]    # shape [B, 1]
            time_idx  = torch.arange(seq_len, device=action.device)[None, :]     # shape [1, T]
            valid = rt[batch_idx, time_idx, action] == 1
            new_mask[batch_idx, time_idx, action] = valid.float()

        #xt_input = self._compose_inputs(xt, new_mask)  # [B,N,2*Dx]

        new_batch = {}
        new_batch['x'] = batch['x']
        new_batch['xc'] = batch['xc']  # [B,Nc,2*Dx]
        new_batch['xt'] = batch['xt']

        new_batch['y'] = batch['y']
        new_batch['yc'] = batch['yc']
        new_batch['yt'] = batch['yt']

        new_batch['mask'] = torch.cat([batch['maskc'], new_mask], dim=1)
        new_batch['maskc'] = batch['maskc']
        new_batch['maskt'] = new_mask

        new_batch['r'] = batch['r']
        new_batch['rc'] = batch['rc']
        new_batch['rt'] = batch['rt']
    
        return new_batch, new_mask


class RBFKernel(object):
    def __init__(self, sigma_eps=2e-2, max_length=5.0, max_scale=2.0):
        self.sigma_eps = sigma_eps
        self.max_length = max_length
        self.max_scale = max_scale

    # x: batch_size * num_points * dim  [B,N,Dx=1]
    def __call__(self, x):
        B, N, D = x.shape
        
        informative_masks = []
        for b in range(B):
            num_informative = torch.randint(1, D+1, (1,), device=x.device).item()
            informative_idx = torch.randperm(D, device=x.device)[:num_informative]
            
            mask = torch.ones(D, device=x.device) * 1e6
            mask[informative_idx] = 1.0
            informative_masks.append(mask)

        mask = torch.stack(informative_masks, dim=0)

        rbf = gpytorch.kernels.RBFKernel(batch_shape=torch.Size([B]), ard_num_dims=D)
        scale = gpytorch.kernels.ScaleKernel(rbf, batch_shape=torch.Size([B]))

        rbf.to(x.device)
        scale.to(x.device)

        lengthscales = 0.1 + (self.max_length - 0.1) * torch.rand(B, D, device=x.device)   # [B, D]
        outputscales = 0.5 + (self.max_scale - 0.5) * torch.rand(B, device=x.device)      # [B]

        lengthscales = lengthscales * mask

        rbf.lengthscale = lengthscales.view(B, 1, D) 
        scale.outputscale = outputscales

        # Compute covariance for each batch
        eye = torch.eye(N, device=x.device).expand(B, N, N)
        cov = scale(x, x).to_dense() + (self.sigma_eps**2) * eye

        cov = torch.linalg.cholesky(cov)

        return cov
    
class MaternKernel(object):
    def __init__(self, sigma_eps=2e-2, max_length=5.0, max_scale=2.0, nu=2.5):
        """
        sigma_eps: additive observation noise
        max_length: max lengthscale per feature
        max_scale: max output scale
        nu: smoothness parameter of Matern kernel (1.5, 2.5, etc.)
        """
        self.sigma_eps = sigma_eps
        self.max_length = max_length
        self.max_scale = max_scale
        self.nu = nu

    def __call__(self, x: torch.Tensor):
        """
        x: [B, N, D] batch of inputs
        returns: [B, N, N] lower-triangular Cholesky matrices
        """
        B, N, D = x.shape

        # Generate per-batch informative feature masks
        informative_masks = []
        for b in range(B):
            num_informative = torch.randint(1, D + 1, (1,), device=x.device).item()
            informative_idx = torch.randperm(D, device=x.device)[:num_informative]

            mask = torch.ones(D, device=x.device) * 1e6
            mask[informative_idx] = 1.0
            informative_masks.append(mask)
        mask = torch.stack(informative_masks, dim=0)  # [B, D]

        # Define Matern kernel with ARD
        matern = gpytorch.kernels.MaternKernel(nu=self.nu, batch_shape=torch.Size([B]), ard_num_dims=D)
        scale = gpytorch.kernels.ScaleKernel(matern, batch_shape=torch.Size([B]))
        matern.to(x.device)
        scale.to(x.device)

        # Sample lengthscales and output scales per batch
        lengthscales = 0.1 + (self.max_length - 0.1) * torch.rand(B, D, device=x.device)  # [B, D]
        outputscales = 0.5 + (self.max_scale - 0.5) * torch.rand(B, device=x.device)      # [B]
        lengthscales = lengthscales * mask

        matern.lengthscale = lengthscales.view(B, 1, D)
        scale.outputscale = outputscales

        # Compute covariance and add noise
        eye = torch.eye(N, device=x.device).expand(B, N, N)
        cov = scale(x, x).to_dense() + (self.sigma_eps**2) * eye

        # Cholesky decomposition
        chol = torch.linalg.cholesky(cov)

        return chol