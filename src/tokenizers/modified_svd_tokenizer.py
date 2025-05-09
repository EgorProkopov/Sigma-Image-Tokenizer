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

        self.cls_token = nn.Parameter(torch.randn(1, 1, embedding_dim))
        self.positional_encoding = PositionalEncoding(embedding_dim)


    def _get_raw_tokens(self, x):
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

    def _add_cls_token(self, tokens):
        batch_size = tokens.shape[0]
        cls_token = self.cls_token.expand(batch_size, -1, -1)
        tokens = torch.cat([cls_token, tokens], dim=1)
        return tokens

    def _add_positional_encoding(self, tokens):
        positional_encoding = self.positional_encoding(tokens)
        tokens = tokens + positional_encoding
        return tokens

    def _filter_tokens(self, sorted_weighted_tokens, sorted_sigmas):
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
        raw_tokens = self._get_raw_tokens(x)
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
        tokens = self._add_cls_token(tokens)
        tokens = self._add_positional_encoding(tokens)

        return tokens


# class MSVDSigmoidGatingTokenizer(MSVDNoScorerTokenizer):
#     def __init__(
#             self,
#             in_channels=3,
#             pixel_unshuffle_scale_factors=[2, 2, 2, 2],
#             dispersion=0.9,
#             embedding_dim=768
#     ):
#         super().__init__(in_channels, pixel_unshuffle_scale_factors, dispersion, embedding_dim)
#
#         self.mlp_scorer = nn.Sequential(
#             nn.Linear(
#                 in_features=embedding_dim,
#                 out_features=128,
#             ),
#             nn.InstanceNorm1d(128),
#             nn.LeakyReLU(),
#             nn.Linear(
#                 in_features=128, out_features=1
#             ),
#             # nn.ReLU()
#         )
#
#     def forward(self, x):
#         raw_tokens = self._get_raw_tokens(x)
#         tokens = self.linear_projection(raw_tokens)
#
#         scores = self.mlp_scorer(tokens)
#         gates = nn.functional.sigmoid(scores)
#         gated_tokens = tokens * gates
#
#         tokens = self._add_cls_token(gated_tokens)
#         tokens = self._add_positional_encoding(tokens)
#
#         return {"tokens": tokens, "scores": scores}


class MSVDSigmoidGatingTokenizer(MSVDNoScorerTokenizer):
    def __init__(
        self,
        in_channels: int = 3,
        pixel_unshuffle_scale_factors: list = [2, 2, 2, 2],
        embedding_dim: int = 768,
        selection_mode: str = "full",
        top_k: int = None,
        dispersion: float = None,
    ):
        super().__init__(in_channels, pixel_unshuffle_scale_factors, dispersion, embedding_dim)

        modes = {"full", "top-k", "dispersion"}
        assert selection_mode in modes, f"selection_mode must be one of {modes}"
        if selection_mode == "top-k":
            assert top_k is not None and top_k > 0, "top-k value must be positive"

        self.selection_mode = selection_mode
        self.top_k = top_k

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
            # nn.ReLU()
        )

    def _filter_tokens(self, tokens: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        gates = torch.sigmoid(scores)              # [B, N]
        gated = tokens * gates.unsqueeze(-1)       # [B, N, E]
        B, N, E = tokens.shape

        if self.training or self.selection_mode == "full":
            return gated

        if self.selection_mode == "top-k":
            k = min(self.top_k, N)
            idx = scores.topk(k, dim=1).indices    # [B, k]
            mask = torch.zeros_like(scores)
            mask.scatter_(1, idx, 1.0)
            return tokens * mask.unsqueeze(-1)

        gates_sorted, idx_sorted = gates.sort(dim=1, descending=True)  # [B, N]
        cumsum = gates_sorted.cumsum(dim=1)                                # [B, N]
        total = gates_sorted.sum(dim=1, keepdim=True)                   # [B, 1]
        thresh = self.dispersion * total                                # [B, 1]
        keep_sorted = cumsum <= thresh                                    # [B, N]
        mask = torch.zeros_like(gates)
        mask.scatter_(1, idx_sorted, keep_sorted.float())              # [B, N]
        return tokens * mask.unsqueeze(-1)

    def forward(self, x: torch.Tensor):
        raw_tokens = self._get_raw_tokens(x)           # [B, N, C]
        tokens = self.linear_projection(raw_tokens)    # [B, N, E]

        scores = self.mlp_scorer(tokens).squeeze(-1)   # [B, N]

        filtered = self._filter_tokens(tokens, scores) # [B, N, E]

        out = self._add_cls_token(filtered)            # [B, N+1, E]
        out = self._add_positional_encoding(out)

        return {"tokens": out, "scores": scores}

