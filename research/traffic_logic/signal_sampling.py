"""Paired probes using the unchanged official lane-conditioned sampling loop."""
import hashlib

import torch
from torch import nn

from research.traffic_logic.signal_lane_adapter import LaneSignalResidual


def _rng_digest(device):
    states = [torch.random.get_rng_state()]
    if device.type == 'cuda':
        states.append(torch.cuda.get_rng_state(device))
    digest = hashlib.sha256()
    for state in states:
        digest.update(state.cpu().numpy().tobytes())
    return digest.hexdigest()


class ObservedSignalDiT(nn.Module):
    """Observe every official CFG call, injecting only on conditional calls."""

    def __init__(self, base, adapter, condition, device, progress):
        super().__init__()
        self.base, self.adapter = base, adapter
        self.condition, self.device = condition, device
        self.progress = progress
        self.trace = {'conditional_calls': 0, 'unconditional_calls': 0,
                      'injection_calls': 0, 'lane_timesteps': [], 'rng_call_digests': []}
        self.initial_agent_noise = None

    def forward(self, xa, xl, data, ta, tl, unconditional=False):
        if not torch.isfinite(xa).all() or not torch.isfinite(xl).all():
            raise AssertionError('Non-finite sampler input')
        self.trace['rng_call_digests'].append(_rng_digest(self.device))
        if unconditional:
            self.trace['unconditional_calls'] += 1
        else:
            self.trace['conditional_calls'] += 1
            self.trace['lane_timesteps'].append(int(tl[0]))
            if self.initial_agent_noise is None:
                self.initial_agent_noise = xa.detach().cpu().clone()
            torch.testing.assert_close(xl[:, 0], data['lane'].latents, rtol=0, atol=0)
        if self.condition is not None and not unconditional:
            with LaneSignalResidual(self.base.lane_embedder, self.adapter, *self.condition) as hook:
                output = self.base(xa, xl, data, ta, tl, unconditional=False)
            if hook.calls != 1:
                raise AssertionError('Expected exactly one conditional signal injection')
            self.trace['injection_calls'] += hook.calls
            self.progress['model_condition_injected'] = True
        else:
            output = self.base(xa, xl, data, ta, tl, unconditional=unconditional)
        if not all(torch.isfinite(value).all() for value in output):
            raise AssertionError('Non-finite DiT prediction')
        return output


@torch.no_grad()
def sample_condition_variants(ldm, data, adapter, conditions, seed=20261013, progress=None, on_variant=None):
    """Run named conditions through the same official loop and paired RNG stream."""
    conditions = list(conditions)
    labels = [label for label, _ in conditions]
    if not labels or len(set(labels)) != len(labels) or any(not isinstance(x, str) or not x for x in labels):
        raise ValueError('Expected unique nonempty variant names')
    ldm.eval().requires_grad_(False)
    adapter.eval().requires_grad_(False)
    device = next(ldm.parameters()).device
    data = data.clone().to(device)
    if data.num_graphs != 1:
        raise ValueError('This paired probe requires one scene')
    base = ldm.model
    versions = tuple(p._version for p in ldm.parameters())
    adapter_versions = tuple(p._version for p in adapter.parameters())
    state_keys = tuple(ldm.state_dict())
    shapes = (data['agent'].x.unsqueeze(1).shape, data['lane'].x.unsqueeze(1).shape)
    fixed = data['lane'].latents
    outputs, traces, initial = {}, {}, {}
    report = {} if progress is None else progress
    report.update({'sampling_mode': 'official lane_conditioned', 'seed': seed,
                   'diffusion_steps': ldm.n_timesteps, 'quality_improvement_verified': False,
                   'weights_updated': False, 'variants': traces})
    devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == 'cuda' else []
    for label, selected in conditions:
        selected = None if selected is None else tuple(value.to(device) for value in selected)
        if on_variant is not None:
            on_variant(label)
        observed = ObservedSignalDiT(base, adapter, selected, device, report).eval()
        traces[label] = observed.trace
        try:
            ldm.model = observed
            with torch.random.fork_rng(devices=devices):
                torch.random.default_generator.manual_seed(seed)
                if device.type == 'cuda':
                    torch.cuda.default_generators[devices[0]].manual_seed(seed)
                agents, lanes, chain = ldm.p_sample_loop(
                    *shapes, data, device=device, mode='lane_conditioned', return_diffusion_chain=True)
                observed.trace['rng_final_digest'] = _rng_digest(device)
            if len(chain) != ldm.n_timesteps + 1:
                raise AssertionError('Incomplete official diffusion chain')
            for step_agents, step_lanes in chain:
                if not torch.isfinite(step_agents).all() or not torch.isfinite(step_lanes).all():
                    raise AssertionError('Non-finite saved diffusion state')
                torch.testing.assert_close(step_lanes[:, 0], fixed, rtol=0, atol=0)
            observed.trace['fixed_road_latents_verified'] = True
            observed.trace['finite_saved_chain_verified'] = True
            observed.trace['final_agent_clip_fraction'] = float((agents.abs() >= ldm.cfg.model.diffusion_clip).float().mean())
            observed.trace['initial_agent_noise_sha256'] = hashlib.sha256(observed.initial_agent_noise.numpy().tobytes()).hexdigest()
            if (observed.trace['conditional_calls'] != ldm.n_timesteps
                    or observed.trace['unconditional_calls'] != ldm.n_timesteps
                    or observed.trace['injection_calls'] != (ldm.n_timesteps if selected is not None else 0)
                    or observed.trace['lane_timesteps'] != list(range(ldm.n_timesteps - 1, -1, -1))):
                raise AssertionError('Official CFG call count or schedule changed')
            outputs[label] = (agents.cpu(), lanes.cpu())
            traces[label] = observed.trace
            initial[label] = observed.initial_agent_noise
            report['completed_variants'] = list(outputs)
        finally:
            ldm.model = base
    anchor_label = labels[0]
    anchor = traces[anchor_label]
    for label in traces:
        torch.testing.assert_close(initial[label], initial[anchor_label], rtol=0, atol=0)
        for key in ('rng_call_digests', 'rng_final_digest'):
            if traces[label][key] != anchor[key]:
                raise AssertionError('Paired Torch random streams differ: ' + label)
    unchanged = (tuple(p._version for p in ldm.parameters()) == versions
                 and tuple(p._version for p in adapter.parameters()) == adapter_versions
                 and all(p.grad is None for p in ldm.parameters())
                 and all(p.grad is None for p in adapter.parameters())
                 and tuple(ldm.state_dict()) == state_keys)
    if not unchanged:
        raise AssertionError('Frozen pretrained model or adapter changed')
    report.update({'paired_torch_noise_verified': True, 'pretrained_parameters_unchanged': True,
                   'adapter_parameters_unchanged': True, 'full_sampling_run': True})
    return outputs, report


@torch.no_grad()
def sample_paired(ldm, data, adapter, condition, seed=20261013, progress=None, on_variant=None):
    """E7 compatibility: original repeat, adapter STOP and all-active synthetic GO."""
    features, mask, geometry, active = condition
    changed = features.clone()
    changed[active, :9] = 0
    changed[active, 6] = 1
    conditions = [('baseline', None), ('baseline_repeat', None),
                  ('stop', condition), ('go', (changed, mask, geometry, active))]
    outputs, report = sample_condition_variants(ldm, data, adapter, conditions, seed, progress, on_variant)
    numerical_delta = float((outputs['baseline'][0] - outputs['baseline_repeat'][0]).abs().max())
    threshold = max(1e-5, 5 * numerical_delta)
    response = {label: float((outputs[a][0] - outputs[b][0]).abs().max())
                for label, a, b in [('adapter_vs_baseline', 'stop', 'baseline'), ('go_vs_stop', 'go', 'stop')]}
    report.update({'paired_torch_noise_verified': True, 'pretrained_parameters_unchanged': True,
                   'adapter_parameters_unchanged': True, 'full_sampling_run': True,
                   'baseline_repeat_agent_latent_max_abs': numerical_delta,
                   'exploratory_latent_response_threshold': threshold,
                   'agent_latent_differences': response,
                   'exploratory_response_above_baseline_repeat': {k: v > threshold for k, v in response.items()},
                   'response_scope': 'one paired seed; one baseline repeat; descriptive, not statistical validation'})
    return outputs, report
