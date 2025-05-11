import pywt
import torch
import numpy as np
import torch.nn as nn


class WaveletTokenizer(nn.Module):
    def __init__(self, wavelet='haar', bit_planes=4, final_threshold=2 ** -3):
        super().__init__()
        self.wavelet = wavelet
        self.bit_planes = bit_planes
        self.final_threshold = final_threshold
        self.tokens = {
            "Group4x4": 0,
            "Group2x2": 1,
            "Insignificant": 2,
            "NowSignificantPos": 3,
            "NowSignificantNeg": 4,
            "NextAccuracy0": 5,
            "NextAccuracy1": 6
        }
        self.vocab_size = 7

    def forward(self, image: torch.Tensor):
        coeffs = pywt.wavedec2(image.cpu().numpy(), self.wavelet, level=self.bit_planes)
        coeff_arr, slices = pywt.coeffs_to_array(coeffs, axes=(0, 1))

        # Initialize threshold based on max absolute value
        T = 2 ** (np.ceil(np.log2(np.max(np.abs(coeff_arr)))) - 1)

        tokens_sequence = []
        approx_coeffs = np.zeros_like(coeff_arr)

        for bp in range(self.bit_planes):
            self._scan_bit_plane(coeff_arr, approx_coeffs, tokens_sequence, T)
            T /= 2  # update threshold

            if T < self.final_threshold:
                break

        tokens_tensor = torch.tensor(tokens_sequence, dtype=torch.long)

        return tokens_tensor

    def _scan_bit_plane(self, coeff_arr, approx_coeffs, tokens_sequence, T):
        M, N = coeff_arr.shape

        for i in range(0, M, 4):
            for j in range(0, N, 4):
                block = coeff_arr[i:i+4, j:j+4]

                if np.all(np.abs(block) < T):
                    tokens_sequence.append(self.tokens["Group4x4"])
                else:
                    for ii in range(i, min(i+4, M), 2):
                        for jj in range(j, min(j+4, N), 2):
                            subblock = coeff_arr[ii:ii+2, jj:jj+2]
                            subblock_approx = approx_coeffs[ii:ii+2, jj:jj+2]

                            if np.all(np.abs(subblock) < T):
                                tokens_sequence.append(self.tokens["Group2x2"])
                            else:
                                for iii in range(ii, min(ii+2, M)):
                                    for jjj in range(jj, min(jj+2, N)):
                                        coef = coeff_arr[iii, jjj]
                                        approx_coef = subblock_approx[iii - ii, jjj - jj]

                                        if np.abs(coef) >= 2*T:
                                            token = "NextAccuracy1" if np.abs(coef) > np.abs(approx_coef) else "NextAccuracy0"
                                            tokens_sequence.append(self.tokens[token])
                                            # обновление approx_coeffs
                                            delta = T/4 if token == "NextAccuracy1" else -T/4
                                            approx_coeffs[iii, jjj] += np.sign(coef) * delta
                                        elif T <= np.abs(coef) < 2*T:
                                            now_token = "NowSignificantPos" if coef > 0 else "NowSignificantNeg"
                                            tokens_sequence.append(self.tokens[now_token])
                                            # устанавливаем приближение в середину сегмента
                                            approx_coeffs[iii, jjj] = np.sign(coef) * 1.5 * T

                                            next_token = "NextAccuracy1" if np.abs(coef) > 1.5*T else "NextAccuracy0"
                                            tokens_sequence.append(self.tokens[next_token])
                                            delta = T/4 if next_token == "NextAccuracy1" else -T/4
                                            approx_coeffs[iii, jjj] += np.sign(coef) * delta
                                        else:
                                            tokens_sequence.append(self.tokens["Insignificant"])

