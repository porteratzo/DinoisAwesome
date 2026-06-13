"""Unit tests for dinoisawesome.instance_detection.

All tests operate on synthetic tensors — no model weights are loaded.
"""

import pytest
import torch
import torch.nn.functional as F

from dinoisawesome.instance_detection import (
    compute_density_map,
    compute_exemplar_features,
    extract_peaks,
)

# ---------------------------------------------------------------------------
# compute_exemplar_features
# ---------------------------------------------------------------------------


class TestComputeExemplarFeatures:
    def _tokens(self, n: int = 100, c: int = 64) -> torch.Tensor:
        return F.normalize(torch.randn(n, c), p=2, dim=-1)

    def test_mean_output_shape(self):
        out = compute_exemplar_features(self._tokens(), mode="mean")
        assert out.shape == (1, 64)

    def test_mean_output_is_l2_normalised(self):
        out = compute_exemplar_features(self._tokens(), mode="mean")
        assert torch.allclose(out.norm(dim=-1), torch.ones(1), atol=1e-5)

    def test_kmeans_output_shape(self):
        out = compute_exemplar_features(self._tokens(), mode="kmeans", k=4)
        assert out.shape == (4, 64)

    def test_kmeans_output_is_l2_normalised(self):
        out = compute_exemplar_features(self._tokens(), mode="kmeans", k=4)
        assert torch.allclose(out.norm(dim=-1), torch.ones(4), atol=1e-5)

    def test_mean_of_identical_tokens_equals_that_token(self):
        # Mean of N copies of a unit vector should equal that vector.
        v = F.normalize(torch.randn(1, 64), p=2, dim=-1)
        tokens = v.expand(50, -1)
        out = compute_exemplar_features(tokens, mode="mean")
        assert torch.allclose(out, v, atol=1e-5)


# ---------------------------------------------------------------------------
# compute_density_map
# ---------------------------------------------------------------------------


class TestComputeDensityMap:
    H, W, C = 8, 8, 32

    def _make(self):
        tokens = F.normalize(torch.randn(self.H * self.W, self.C), p=2, dim=-1)
        feats = F.normalize(torch.randn(1, self.C), p=2, dim=-1)
        return tokens, feats

    def test_output_shape(self):
        tokens, feats = self._make()
        dm = compute_density_map(tokens, feats, self.H, self.W, threshold=0.0)
        assert dm.shape == (self.H, self.W)

    def test_all_zeros_when_threshold_is_huge(self):
        tokens, feats = self._make()
        dm = compute_density_map(tokens, feats, self.H, self.W, threshold=999.0)
        assert dm.eq(0).all()

    def test_nonnegative_output(self):
        tokens, feats = self._make()
        dm = compute_density_map(tokens, feats, self.H, self.W, threshold=0.0)
        assert (dm >= 0).all()

    def test_perfect_match_is_global_maximum(self):
        """A query token identical to the exemplar must yield the highest density."""
        feats = F.normalize(torch.randn(1, self.C), p=2, dim=-1)
        tokens = F.normalize(torch.randn(self.H * self.W, self.C), p=2, dim=-1)
        row, col = 3, 5
        tokens[row * self.W + col] = feats[0]
        dm = compute_density_map(tokens, feats, self.H, self.W, threshold=0.0)
        assert dm[row, col].item() == pytest.approx(dm.max().item(), abs=1e-5)

    def test_multi_descriptor_averages_similarities(self):
        """K=2 descriptors: clamp(mean(raw_sims), 0), NOT mean(clamp(raw_sims)).

        The pipeline averages cosine similarities across descriptors first,
        then applies the ReLU-style clamp — so tokens with one positive and one
        negative similarity can still survive if their average is positive.
        """
        tokens = F.normalize(torch.randn(self.H * self.W, self.C), p=2, dim=-1)
        f1 = F.normalize(torch.randn(1, self.C), p=2, dim=-1)
        f2 = F.normalize(torch.randn(1, self.C), p=2, dim=-1)
        feats_combined = torch.cat([f1, f2], dim=0)  # (2, C)

        dm_combined = compute_density_map(tokens, feats_combined, self.H, self.W, threshold=0.0)

        # Manually replicate: average raw sims, then clamp
        sim1 = (tokens @ f1.T).squeeze()  # (H*W,)
        sim2 = (tokens @ f2.T).squeeze()  # (H*W,)
        expected = torch.clamp(((sim1 + sim2) / 2.0).reshape(self.H, self.W), min=0.0)

        assert torch.allclose(dm_combined, expected, atol=1e-5)


# ---------------------------------------------------------------------------
# extract_peaks
# ---------------------------------------------------------------------------


class TestExtractPeaks:
    def test_no_peaks_on_zero_map(self):
        dm = torch.zeros(8, 8)
        peaks = extract_peaks(dm, kernel_size=3, min_peak_threshold=0.1)
        assert len(peaks) == 0

    def test_single_peak_detected(self):
        dm = torch.zeros(8, 8)
        dm[3, 5] = 1.0  # row=3, col=5
        peaks = extract_peaks(dm, kernel_size=3, min_peak_threshold=0.1)
        assert len(peaks) >= 1
        found = any(p[0].item() == 5 and p[1].item() == 3 for p in peaks)
        assert found, f"Expected peak at (x=5, y=3), got {peaks.tolist()}"

    def test_min_threshold_suppresses_weak_peaks(self):
        dm = torch.zeros(8, 8)
        dm[1, 1] = 0.05  # weak — below threshold
        dm[6, 6] = 0.50  # strong
        peaks = extract_peaks(dm, kernel_size=3, min_peak_threshold=0.1)
        assert len(peaks) == 1
        assert peaks[0, 0].item() == 6  # x = col
        assert peaks[0, 1].item() == 6  # y = row

    def test_output_column_order_is_x_then_y(self):
        """peaks[:, 0] must be x (col) and peaks[:, 1] must be y (row)."""
        dm = torch.zeros(10, 10)
        dm[2, 7] = 1.0  # row=2, col=7  → x=7, y=2
        peaks = extract_peaks(dm, kernel_size=3, min_peak_threshold=0.1)
        assert len(peaks) >= 1
        assert peaks[0, 0].item() == 7  # x = col
        assert peaks[0, 1].item() == 2  # y = row

    def test_two_separated_peaks(self):
        dm = torch.zeros(10, 10)
        dm[1, 1] = 0.8
        dm[8, 8] = 0.9
        peaks = extract_peaks(dm, kernel_size=3, min_peak_threshold=0.1)
        xy = {(p[0].item(), p[1].item()) for p in peaks}
        assert (1, 1) in xy
        assert (8, 8) in xy

    def test_returns_integer_coordinates(self):
        dm = torch.zeros(8, 8)
        dm[4, 4] = 1.0
        peaks = extract_peaks(dm, kernel_size=3, min_peak_threshold=0.1)
        assert peaks.dtype in (torch.int32, torch.int64, torch.long)

    def test_adjacent_peaks_merged_by_large_kernel(self):
        """With a large NMS kernel, neighbouring activations should merge to one peak."""
        dm = torch.zeros(10, 10)
        dm[5, 4] = 0.9
        dm[5, 5] = 0.8  # adjacent — should be suppressed by 5×5 NMS
        peaks = extract_peaks(dm, kernel_size=5, min_peak_threshold=0.1)
        # Only the higher value should survive
        assert len(peaks) == 1
        assert peaks[0, 0].item() == 4  # x = col of max
        assert peaks[0, 1].item() == 5  # y = row of max
