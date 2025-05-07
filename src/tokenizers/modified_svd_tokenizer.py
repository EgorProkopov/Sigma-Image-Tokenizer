import torch
import torch.nn as nn

from src.tokenizers.positional_encoding import PositionalEncoding


class MSVDNoScorerTokenizer(nn.Module):
    def __init__(
            self,
            in_channels=3,
            pixel_unshuffle_scale_factors=[2, 2, 2, 2],
            dispersion=0.9,
            embedding_dim=768
    ):
        super().__init__()

        self.in_channels = in_channels
        self.pixel_unshuffle_scale_factors = pixel_unshuffle_scale_factors
        self.dispersion=dispersion

        self.u_feature_extractor = nn.ModuleList()
        current_channels = self.in_channels
        for scale in self.pixel_unshuffle_scale_factors:
            self.u_feature_extractor.append(
                nn.Conv2d(
                    in_channels=current_channels,
                    out_channels=current_channels,
                    kernel_size=3, padding=1, stride=1
                ),
            )
            self.u_feature_extractor.append(nn.BatchNorm2d(current_channels))
            self.u_feature_extractor.append(nn.LeakyReLU())
            self.u_feature_extractor.append(nn.PixelUnshuffle(downscale_factor=scale))
            current_channels = current_channels * (scale ** 2)
        self.u_feature_extractor = nn.Sequential(*self.u_feature_extractor)

        self.v_feature_extractor = nn.ModuleList()
        current_channels = self.in_channels
        for scale in self.pixel_unshuffle_scale_factors:
            self.v_feature_extractor.append(
                nn.Conv2d(
                    in_channels=current_channels,
                    out_channels=current_channels,
                    kernel_size=3, padding=1, stride=1
                )
            )
            self.v_feature_extractor.append(nn.BatchNorm2d(current_channels))
            self.v_feature_extractor.append(nn.LeakyReLU())
            self.v_feature_extractor.append(nn.PixelUnshuffle(downscale_factor=scale))
            current_channels = current_channels * (scale ** 2)
        self.v_feature_extractor = nn.Sequential(*self.v_feature_extractor)

        self.projection_u = nn.Conv2d(
            in_channels=current_channels,
            out_channels=current_channels,
            kernel_size=1, stride=1, padding=0, bias=False
        )
        self.projection_v = nn.Conv2d(
            in_channels=current_channels,
            out_channels=current_channels,
            kernel_size=1, stride=1, padding=0, bias=False
        )

        self.linear_projection = nn.Linear(
            in_features=current_channels * 2,
            out_features=embedding_dim
        )

        self.mlp_scorer = nn.Sequential(
            nn.Linear(
                in_features=embedding_dim,
                out_features=128,
            ),
            nn.InstanceNorm1d(128),
            nn.LeakyReLU(),
            nn.Linear(
                in_features=128, out_features=1
            ),
            nn.ReLU()
        )

        self.cls_token = nn.Parameter(torch.randn(1, 1, embedding_dim))
        self.positional_encoding = PositionalEncoding(embedding_dim)


    def __get_raw_tokens(self, x):
        raw_u = self.u_feature_extractor(x)
        raw_v = self.v_feature_extractor(x)

        projected_u = self.projection_u(raw_u)
        projected_v = self.projection_v(raw_v)

        projected_u = projected_u.permute(0, 2, 3, 1).contiguous()
        projected_v = projected_v.permute(0, 2, 3, 1).contiguous()

        B, H, W, C = projected_u.shape
        tokens_u = torch.reshape(projected_u, (B, H * W, C))
        tokens_v = torch.reshape(projected_v, (B, H * W, C))

        raw_tokens = torch.cat([tokens_u, tokens_v], dim=-1)
        return raw_tokens

    def __add_cls_token(self, tokens):
        batch_size = tokens.shape[0]
        cls_token = self.cls_token.expand(batch_size, -1, -1)
        tokens = torch.cat([cls_token, tokens], dim=1)
        return tokens

    def __add_positional_encoding(self, tokens):
        positional_encoding = self.positional_encoding(tokens)
        tokens = tokens + positional_encoding
        return tokens

    def __filter_tokens(self, sorted_weighted_tokens, sorted_sigmas):
        batch_size, n_tokens, raw_tokens_dim = sorted_weighted_tokens.shape

        sorted_sigmas_squeezed = sorted_sigmas.squeeze(dim=-1)
        sigma_sum = sorted_sigmas_squeezed.sum(dim=1, keepdim=True)
        threshold = self.dispersion * sigma_sum.unsqueeze(-1)

        cumsum_sigma = sorted_sigmas.cumsum(dim=1)
        keep_mask = cumsum_sigma <= threshold

        lengths = keep_mask.sum(dim=1)

        max_len = lengths.max().item()
        padded = sorted_weighted_tokens.new_zeros(batch_size, max_len, raw_tokens_dim)
        for b in range(batch_size):
            L = lengths[b]
            padded[b, :L] = sorted_weighted_tokens[b, :L]
        return padded, lengths

    def forward(self, x):
        raw_tokens = self.__get_raw_tokens(x)
        # sigmas = self.mlp_scorer(raw_tokens)
        # weighted_tokens = raw_tokens * sigmas

        # # Sort in sigma-ascending order
        # sort_idxs = torch.argsort(sigmas, dim=1)
        # sorted_sigmas = torch.gather(
        #     sigmas,  # (B, N, 1)
        #     dim=1,
        #     index=sort_idxs  # (B, N, 1)
        # )
        # sorted_weighted_tokens = torch.gather(
        #     weighted_tokens,  # (B, N, 2C)
        #     dim=1,
        #     index=sort_idxs.expand(-1, -1, weighted_tokens.size(2))
        # )

        # sorted_sigmas = sigmas
        # sorted_weighted_tokens = weighted_tokens

        # sorted_weighted_tokens, lengths = self.__filter_tokens(sorted_weighted_tokens, sorted_sigmas)
        # print(lengths.max().item())


        tokens = self.linear_projection(raw_tokens)
        tokens = self.__add_cls_token(tokens)
        tokens = self.__add_positional_encoding(tokens)

        return tokens

