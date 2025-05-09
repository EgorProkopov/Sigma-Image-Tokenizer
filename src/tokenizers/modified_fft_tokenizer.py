import torch
import torch.nn as nn

from src.tokenizers.positional_encoding import PositionalEncoding


class FFTLowFreqFilter(nn.Module):
    """
    Module to compute 2D FFT of a 3-channel image batch and extract low-frequency components.

    Input:
        x: Tensor of shape [B, 3, W, H]
    Output:
        Tensor of shape [B, 6, filter_size, filter_size] containing real and imaginary
        parts of the low-frequency region for each channel.
    """
    def __init__(self, filter_size: int = 128, energy_ratio: float = 0.900):
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

    def forward(self, x: torch.Tensor) -> dict:
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

        return {"tensor": low_freq, "filter_size": filter_size}


class MFFTTokenizer(nn.Module):
    def __init__(
            self,
            in_channels: int = 3,
            pixel_unshuffle_scale_factors: list = [2, 2, 2, 2],
            embedding_dim: int = 768,
            filter_size: int = 128,
            energy_ratio: float = 0.900
    ):
        super().__init__()

        self.low_freq_filter = FFTLowFreqFilter(filter_size=filter_size, energy_ratio=energy_ratio)

        self.in_channels = in_channels
        self.pixel_unshuffle_scale_factors = pixel_unshuffle_scale_factors
        self.embedding_dim = embedding_dim

        feature_extractor_layers = nn.ModuleList()
        current_channels = self.in_channels * 2
        for scale in self.pixel_unshuffle_scale_factors:
            feature_extractor_layers.append(
                nn.Conv2d(
                    in_channels=current_channels,
                    out_channels=current_channels,
                    kernel_size=3, padding=1, stride=1
                )
            )
            feature_extractor_layers.append(nn.BatchNorm2d(current_channels))
            feature_extractor_layers.append(nn.LeakyReLU())
            feature_extractor_layers.append(nn.PixelUnshuffle(downscale_factor=scale))
            current_channels = current_channels * (scale ** 2)

        self.feature_extractor = nn.Sequential(*feature_extractor_layers)

        self.projection = nn.Conv2d(
            in_channels=current_channels,
            out_channels=current_channels,
            kernel_size=1, stride=1, padding=0, bias=False
        )

        self.linear_projection = nn.Linear(
            in_features=current_channels,
            out_features=embedding_dim
        )

        self.cls_token = nn.Parameter(torch.randn(1, 1, embedding_dim))
        self.positional_encoding = PositionalEncoding(embedding_dim)

    def forward(self, x) -> dict:
        low_freq_output = self.low_freq_filter(x)  # [B, 6, fs, fs]

        low_freq_x = low_freq_output["tensor"]
        filter_size = low_freq_output["filter_size"]

        features = self.feature_extractor(low_freq_x)  # [B, 6 * (4**N), fs / (2**N), fs / (2**N)]
        features = self.projection(features)  # [B, 6 * (4**N), fs / (2**N), fs / (2**N)]
        features = features.permute(0, 2, 3, 1).contiguous()  # [B, fs / (2**N), fs / (2**N), 6 * (4**N)]

        B, H, W, C = features.shape
        raw_tokens = torch.reshape(features, (B, H * W, C))

        tokens = self.linear_projection(raw_tokens)
        tokens = self._add_cls_token(tokens)
        tokens = self._add_positional_encoding(tokens)

        return {"tokens": tokens, "filter_size": filter_size}
