import unittest

import torch

from research.traffic_logic.signal_lane_adapter import SignalLaneAdapter, LaneSignalResidual


def inputs():
    features = torch.zeros(2, 11)
    features[:, 4] = 1
    features[:, 9:] = torch.tensor([[0.2, 0.3], [-0.3, 0.4]])
    geometry = torch.arange(24, dtype=torch.float32).reshape(3, 4, 2) / 20 - 0.5
    masks = torch.zeros(2, 3, 4, dtype=torch.bool)
    masks[0, 1, 1:3] = True
    masks[1, 2, 3] = True
    return features, masks, geometry, torch.tensor([True, False])


class SignalAdapterTest(unittest.TestCase):
    def learned_adapter(self):
        adapter = SignalLaneAdapter(8, hidden_dim=4)
        with torch.no_grad():
            adapter.input.weight.fill_(0.1)
            adapter.input.bias.fill_(0.5)
            adapter.output.weight.fill_(0.1)
            adapter.output.bias.fill_(0.25)
        return adapter

    def test_zero_initialized_projection_receives_learning_gradient(self):
        torch.manual_seed(3)
        adapter = SignalLaneAdapter(8, hidden_dim=4)
        result = adapter(*inputs())
        self.assertEqual(tuple(result.shape), (3, 8))
        loss = (result - 1).square().mean()
        loss.backward()
        self.assertIsNotNone(adapter.output.weight.grad)
        self.assertGreater(adapter.output.weight.grad.abs().sum().item(), 0)

    def test_initial_residual_is_exactly_zero(self):
        result = SignalLaneAdapter(8)(*inputs())
        torch.testing.assert_close(result, torch.zeros_like(result), rtol=0, atol=0)

    def test_inactive_signals_and_unsupported_rows_ignore_learned_bias(self):
        adapter = self.learned_adapter()
        result = adapter(*inputs())
        torch.testing.assert_close(result[[0, 2]], torch.zeros(2, 8), rtol=0, atol=0)
        self.assertGreater(result[1].abs().sum().item(), 0)
        f, m, g, active = inputs()
        torch.testing.assert_close(adapter(f, m, g, torch.zeros_like(active)), torch.zeros(3, 8), rtol=0, atol=0)

    def test_only_selected_geometry_can_change_residual(self):
        adapter = self.learned_adapter()
        f, m, g, active = inputs()
        original = adapter(f, m, g, active)
        outside = g.clone()
        outside[0] += 20
        outside[1, 0] += 20
        torch.testing.assert_close(adapter(f, m, outside, active), original, rtol=0, atol=0)
        inside = g.clone()
        inside[1, 1] += 1
        self.assertGreater((adapter(f, m, inside, active) - original).abs().max().item(), 0)

    def test_lane_permutation_preserves_routing(self):
        adapter = self.learned_adapter()
        f, m, g, active = inputs()
        perm = torch.tensor([2, 0, 1])
        torch.testing.assert_close(adapter(f, m[:, perm], g[perm], active), adapter(f, m, g, active)[perm])

    def test_overlapping_signals_are_averaged(self):
        adapter = self.learned_adapter()
        f, m, g, active = inputs()
        m[1] = m[0]
        active[:] = True
        both = adapter(f, m, g, active)
        left = adapter(f, m, g, torch.tensor([True, False]))
        right = adapter(f, m, g, torch.tensor([False, True]))
        torch.testing.assert_close(both, (left + right) / 2)

    def test_invalid_mask_dimensions_are_rejected(self):
        f, m, g, active = inputs()
        with self.assertRaises(ValueError):
            SignalLaneAdapter(8)(f, m[:, :2], g, active)

    def test_non_boolean_mask_is_rejected(self):
        f, m, g, active = inputs()
        with self.assertRaises(ValueError):
            SignalLaneAdapter(8)(f, m.float(), g, active)

    def test_empty_signal_set_is_zero(self):
        f, m, g, active = inputs()
        torch.testing.assert_close(self.learned_adapter()(f[:0], m[:0], g, active[:0]), torch.zeros(3, 8), rtol=0, atol=0)

    def test_hook_injects_signal_and_removes_it_after_context(self):
        embedder = torch.nn.Linear(2, 8).requires_grad_(False)
        latent = torch.ones(3, 2)
        baseline = embedder(latent)
        adapter = self.learned_adapter()
        with LaneSignalResidual(embedder, adapter, *inputs()) as hook:
            changed = embedder(latent)
            self.assertGreater((changed - baseline).abs().sum().item(), 0)
            self.assertEqual(hook.calls, 1)
        torch.testing.assert_close(embedder(latent), baseline, rtol=0, atol=0)

    def test_disabled_hook_preserves_baseline(self):
        embedder = torch.nn.Linear(2, 8)
        latent = torch.ones(3, 2)
        baseline = embedder(latent)
        with LaneSignalResidual(embedder, self.learned_adapter(), *inputs(), enabled=False):
            torch.testing.assert_close(embedder(latent), baseline, rtol=0, atol=0)

    def test_hook_allows_learning_through_frozen_embedder(self):
        embedder = torch.nn.Linear(2, 8).requires_grad_(False)
        adapter = SignalLaneAdapter(8)
        with LaneSignalResidual(embedder, adapter, *inputs()):
            result = embedder(torch.ones(3, 2))
            self.assertTrue(result.requires_grad)
            result.square().mean().backward()
        self.assertGreater(adapter.output.weight.grad.abs().sum().item(), 0)
        self.assertTrue(all(p.grad is None for p in embedder.parameters()))

    def test_hook_is_removed_on_exception(self):
        embedder = torch.nn.Linear(2, 8)
        latent = torch.ones(3, 2)
        baseline = embedder(latent)
        with self.assertRaisesRegex(RuntimeError, 'abort'):
            with LaneSignalResidual(embedder, self.learned_adapter(), *inputs()):
                raise RuntimeError('abort')
        torch.testing.assert_close(embedder(latent), baseline, rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
