import torch
import torch.nn as nn


class FFTLowFreqFilter(nn.Module):
    """
    Module to compute 2D FFT of a 3-channel image batch and extract low-frequency components.

    Input:
        x: Tensor of shape [B, 3, W, H]
    Output:
        Tensor of shape [B, 6, filter_size, filter_size] containing real and imaginary
        parts of the low-frequency region for each channel.
    """
    def __init__(self, filter_size: int = 0, energy_ratio: float = 0.900):
        super().__init__()
        self.filter_size = filter_size
        self.energy_ratio = energy_ratio

        self.eps = 1e-8

    def compute_filter_size(self, power_spectrum: torch.Tensor) -> int:
        """
        Compute minimal k such that sum of power_spectrum[..., :k, :k] >= energy_ratio * total_energy
        Uses 2D cumulative sum (integral image) for efficient lookup.

        E: [B, W, H] tensor of spectral energy (magnitude squared)
        Returns: filter_size (int)
        """
        B, W, H = power_spectrum.shape

        cumsum_h = torch.cumsum(power_spectrum, dim=1)        # [B, W, H]
        integral = torch.cumsum(cumsum_h, dim=2) # [B, W, H]

        total_energy = integral[:, -1, -1]       # [B]
        max_k = min(W, H)

        ks = torch.arange(max_k, device=power_spectrum.device)
        b_idx = torch.arange(B, device=power_spectrum.device).unsqueeze(1).expand(B, max_k)
        k_idx = ks.unsqueeze(0).expand(B, max_k)
        cum_energy_k = integral[b_idx, k_idx, k_idx]  # [B, max_k]

        thresholds = total_energy.view(B, 1) * self.energy_ratio
        meets = cum_energy_k >= thresholds
        first_k = torch.where(
            meets.any(dim=1),
            meets.float().argmax(dim=1),
            torch.tensor(max_k - 1, device=power_spectrum.device)
        )  # [B]
        sizes = first_k + 1

        return int(sizes.max().item())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        freq = torch.fft.fft2(x, norm='ortho')  # [B, 3, W, H],

        real = freq.real  # [B, 3, W, H]
        imag = freq.imag  # [B, 3, W, H]

        freq_cat = torch.cat([real, imag], dim=1)  # [B, 6, W, H]

        # log_real = torch.sign(real) * torch.log1p(abs(real) + self.eps)
        # log_imag = torch.sign(imag) * torch.log1p(abs(imag) + self.eps)
        #
        # freq_cat = torch.cat([log_real, log_imag], dim=1)  # [B, 6, W, H]

        B, C, W, H = freq_cat.shape

        filter_size = self.filter_size
        if filter_size == 0:
            power_spectrum = torch.sqrt(real) + torch.sqrt(imag)
            filter_size = self.compute_filter_size(power_spectrum)

        cx, cy = W // 2, H // 2
        half = filter_size // 2
        x_start, y_start = cx - half, cy - half
        x_end, y_end = cx + half, cy + half

        low_freq = freq_cat[:, :, x_start:x_end, y_start:y_end]  # [B, 6, fs, fs]

        return low_freq




