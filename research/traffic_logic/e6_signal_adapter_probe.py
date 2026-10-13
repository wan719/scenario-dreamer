"""E6: verify a signal residual on a previously witnessed L2-style denoiser.

This is a one-scene interface/learning probe, not a traffic-control evaluation.
"""
import argparse
import copy
import gc
import hashlib
import json
import os
import pickle
from pathlib import Path
import subprocess
import tempfile
import traceback

import numpy as np
import torch
from omegaconf import OmegaConf, open_dict
from torch_ema import ExponentialMovingAverage

from nn_modules.ldm import LDM
from research.traffic_logic.signal_lane_adapter import SignalLaneAdapter, LaneSignalResidual


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_reference(reference, arrays, stats, cfg):
    data = reference['data']
    if data.num_graphs != 1 or data['lg_type'].tolist() != [0]:
        raise ValueError('Reference must contain one regular sample')
    for role in ('agent', 'lane'):
        mu = torch.from_numpy(arrays[role + '_mu']).float()
        expected_dim = int(cfg.model[role + '_latent_dim'])
        if mu.ndim != 2 or mu.shape[1] != expected_dim:
            raise ValueError('AE mean latent dimensions differ from model')
        torch.testing.assert_close(data[role].x.cpu(), mu, rtol=0, atol=0)
        mean, std = stats[role + '_latents_mean'], stats[role + '_latents_std']
        if not np.isfinite(np.asarray(mean)).all() or not np.isfinite(np.asarray(std)).all() or np.any(np.asarray(std) <= 0):
            raise ValueError('Invalid latent normalization statistics')
        normalized = (mu - mean) / std
        torch.testing.assert_close(data[role].latents.cpu(), normalized, rtol=1e-6, atol=1e-7)
        expected_shape = (mu.shape[0], 1, expected_dim)
        if tuple(reference['x_' + role].shape) != expected_shape:
            raise ValueError('Wrong denoising input dimensions')
        timestep = reference['t_' + role]
        if timestep.dtype != torch.long or tuple(timestep.shape) != (mu.shape[0],):
            raise ValueError('Wrong timestep shape or dtype')
        if not torch.all((timestep >= 0) & (timestep < cfg.model.n_diffusion_timesteps)):
            raise ValueError('Timestep outside the diffusion schedule')
        for prefix in ('conditional_', 'unconditional_'):
            if tuple(reference[prefix + role].shape) != expected_shape:
                raise ValueError('Wrong saved prediction dimensions')
        if not torch.isfinite(reference['x_' + role]).all():
            raise ValueError('Non-finite denoising input')
    torch.testing.assert_close(reference['x_lane'].cpu(), data['lane'].latents.cpu().unsqueeze(1), rtol=0, atol=0)
    if not torch.all(reference['t_lane'] == 0):
        raise ValueError('Fixed-road reference must use timestep zero')


def run_learning_probe(ldm, reference, condition, steps=5, learning_rate=0.001,
                       seed=20261013, progress=None):
    if steps < 1 or not 0 < learning_rate <= 0.1:
        raise ValueError("Require positive steps and learning rate <= 0.1")
    report = {} if progress is None else progress
    device = next(ldm.parameters()).device
    ldm.eval().requires_grad_(False)
    parameter_versions = tuple(p._version for p in ldm.parameters())
    state_keys = tuple(ldm.state_dict())
    data = reference["data"].to(device)
    xa, xl, ta, tl = [reference[key].to(device) for key in
                       ("x_agent", "x_lane", "t_agent", "t_lane")]
    if not torch.all(tl == 0):
        raise ValueError("This L2 probe requires fixed roads at timestep zero")
    features, masks, geometry, active = [value.to(device) for value in condition]
    condition = (features, masks, geometry, active)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    adapter = SignalLaneAdapter(int(ldm.cfg.model.hidden_dim)).to(device)

    def predict(x_agent, selected_condition=None, unconditional=False):
        if selected_condition is None or unconditional:
            # The signal branch is absent for disabled/CFG-unconditional calls.
            output = ldm.model(x_agent, xl, data, ta, tl, unconditional=unconditional)
            return output, 0
        with LaneSignalResidual(ldm.model.lane_embedder, adapter, *selected_condition) as hook:
            output = ldm.model(x_agent, xl, data, ta, tl, unconditional=False)
        if hook.calls != 1:
            raise AssertionError("Expected one actual lane-embedding injection per DiT call")
        return output, hook.calls

    with torch.no_grad():
        baseline, _ = predict(xa)
        for actual, key in zip(baseline, ("conditional_agent", "conditional_lane")):
            torch.testing.assert_close(actual, reference[key].to(device), rtol=1e-5, atol=1e-6)
        unconditional, _ = predict(xa, unconditional=True)
        for actual, key in zip(unconditional, ("unconditional_agent", "unconditional_lane")):
            torch.testing.assert_close(actual, reference[key].to(device), rtol=1e-5, atol=1e-6)
        initial, zero_calls = predict(xa, condition)
        for before, after in zip(baseline, initial):
            torch.testing.assert_close(after, before, rtol=1e-5, atol=1e-6)
    report.update({
        "baseline_replayed": True, "zero_residual_baseline_verified": True,
        "zero_residual_agent_max_abs": float((initial[0] - baseline[0]).abs().max()),
        "adapter_parameter_count": sum(p.numel() for p in adapter.parameters()),
        "injection_location": "DiT.lane_embedder output, before positional embedding and blocks",
        "step_losses": [], "projection_grad_norms": [], "hook_calls": zero_calls,
        "steps_completed": 0
    })

    # Controlled L2-like objective: q_sample for agents, fixed lane latent, epsilon MSE.
    # This intentionally is not the full official training objective or protocol.
    clean_agents = data["agent"].latents.unsqueeze(1)
    target_noise = torch.randn_like(clean_agents)
    noised_agents = ldm.q_sample(clean_agents, ta, target_noise)
    optimizer = torch.optim.Adam(adapter.parameters(), lr=learning_rate)
    with torch.no_grad():
        prediction, _ = predict(noised_agents)
        loss_before = float((prediction[0] - target_noise).square().mean())

    for step in range(steps):
        optimizer.zero_grad(set_to_none=True)
        prediction, calls = predict(noised_agents, condition)
        loss = (prediction[0] - target_noise).square().mean()
        if not torch.isfinite(loss):
            raise AssertionError("Non-finite probe loss")
        loss.backward()
        gradients = [p.grad for p in adapter.parameters() if p.grad is not None]
        if not gradients or not all(torch.isfinite(value).all() for value in gradients):
            raise AssertionError("Missing or non-finite adapter gradients")
        grad_norm = float(adapter.output.weight.grad.norm())
        if step == 0 and grad_norm <= 0:
            raise AssertionError("Zero-initialized signal projection receives no learning gradient")
        report["projection_grad_norms"].append(grad_norm)
        report["step_losses"].append(float(loss.detach()))
        report["hook_calls"] += calls
        torch.nn.utils.clip_grad_norm_(adapter.parameters(), 1.0)
        optimizer.step()
        report.update({"model_condition_injected": True, "adapter_weights_updated": True,
                       "weights_updated": True, "weights_updated_scope": "adapter only",
                       "steps_completed": step + 1})

    with torch.no_grad():
        trained, calls = predict(xa, condition)
        trained_noisy, extra_calls = predict(noised_agents, condition)
        report["hook_calls"] += calls + extra_calls
        loss_after = float((trained_noisy[0] - target_noise).square().mean())
        changed_features = features.clone()
        changed_features[active, :9] = 0
        changed_features[active, 6] = 1  # Synthetic GO; sensitivity test only.
        changed, calls = predict(xa, (changed_features, masks, geometry, active))
        report["hook_calls"] += calls
        inactive_changed = features.clone()
        inactive_changed[~active] += 0.37
        ignored, calls = predict(xa, (inactive_changed, masks, geometry, active))
        report["hook_calls"] += calls
        torch.testing.assert_close(ignored[0], trained[0], rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(ignored[1], trained[1], rtol=1e-5, atol=1e-6)
        disabled, _ = predict(xa)
        for before, after in zip(baseline, disabled):
            torch.testing.assert_close(after, before, rtol=1e-5, atol=1e-6)
        unconditional_after, _ = predict(xa, condition, unconditional=True)
        for before, after in zip(unconditional, unconditional_after):
            torch.testing.assert_close(after, before, rtol=1e-5, atol=1e-6)
        residual = adapter(*condition)
        support = (masks.any(-1) & active[:, None]).any(0)
        if torch.count_nonzero(residual[~support]):
            raise AssertionError("Residual leaked into unsupported lane rows")
        if not all(torch.isfinite(value).all() for value in (*trained, *changed, residual)):
            raise AssertionError("Non-finite updated predictions")

    unchanged = (tuple(p._version for p in ldm.parameters()) == parameter_versions
                 and all(p.grad is None for p in ldm.parameters())
                 and tuple(ldm.state_dict()) == state_keys)
    if not unchanged:
        raise AssertionError("Frozen pretrained model changed")
    updated_delta = float((trained[0] - baseline[0]).abs().max())
    state_delta = float((changed[0] - trained[0]).abs().max())
    report.update({
        "model_condition_injected": True, "adapter_weights_updated": True,
        "pretrained_parameters_unchanged": unchanged,
        "weights_updated": True, "weights_updated_scope": "adapter only",
        "disabled_branch_baseline_verified": True, "unconditional_branch_baseline_verified": True,
        "inactive_signals_ignored": True, "direct_residual_rows": support.nonzero().flatten().tolist(),
        "loss_before": loss_before, "loss_after": loss_after,
        "loss_improved_in_probe": loss_after < loss_before,
        "updated_agent_max_abs": updated_delta,
        "state_perturbation_agent_max_abs": state_delta,
        "condition_response_verified": updated_delta > 0 and state_delta > 0,
        "counterfactual_scope": "synthetic GO perturbation; no traffic-validity claim",
        "steps": steps, "learning_rate": learning_rate, "seed": seed,
        "objective": "fixed agent epsilon MSE at saved timestep; AE mean latents; fixed roads",
        "full_sampling_run": False, "quality_improvement_verified": False
    })
    artifacts = {
        "adapter_state_dict": {key: value.detach().cpu() for key, value in adapter.state_dict().items()},
        "adapter_hidden_dim": 64, "adapter_output_dim": int(ldm.cfg.model.hidden_dim),
        "reference_agent_noise": baseline[0].cpu(), "updated_agent_noise": trained[0].cpu(),
        "changed_state_agent_noise": changed[0].cpu(), "lane_residual": residual.cpu(),
        "target_noise": target_noise.cpu(), "noised_agents": noised_agents.cpu()
    }
    return report, artifacts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=20261013)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = parser.parse_args()
    baseline = args.baseline_dir.resolve()
    out = Path(tempfile.mkdtemp(prefix="signal_adapter_", dir=baseline.parent))
    report = {"status": "signal_adapter_probe_failed", "output_dir": str(out),
              "baseline_dir": str(baseline), "model_condition_injected": False,
              "weights_updated": False, "quality_improvement_verified": False}
    try:
        if args.steps < 1 or not 0 < args.learning_rate <= 0.1:
            raise ValueError("Invalid probe budget")
        code = Path(__file__).resolve().parents[2]
        for name in ("SCRATCH_ROOT", "DATASET_ROOT"):
            os.environ[name] = str(baseline.parent.parent)
        os.environ["PROJECT_ROOT"] = str(code)
        metadata = json.loads((baseline / "model_base_report.json").read_text())
        if metadata["status"] != "pretrained_regular_forward_verified":
            raise ValueError("Baseline forward is not verified")
        if metadata["scenario_id"] != "637f20cafde22ff8" or metadata["time_index"] != 10:
            raise ValueError("Expected the witnessed record0/t10 scenario")
        for relative, expected in metadata["source_git_blobs"].items():
            actual = subprocess.check_output(["git", "hash-object", relative], cwd=code, text=True).strip()
            if actual != expected:
                raise ValueError("Baseline source changed: " + relative)
        torch.set_num_threads(1)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        if args.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable to this Python environment")
        checkpoint_path = Path(metadata["ldm_checkpoint"]["path"])
        if sha256(checkpoint_path) != metadata["ldm_checkpoint"]["sha256"]:
            raise ValueError("Pretrained checkpoint changed")
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        cfg = copy.deepcopy(ckpt["hyper_parameters"]["cfg"])
        if not OmegaConf.is_config(cfg):
            cfg = OmegaConf.create(cfg)
        with open_dict(cfg.dataset):
            cfg.dataset.num_map_ids = int(metadata["official_num_map_ids"])
        state = {key[len("diff_model."):]: value for key, value in ckpt["state_dict"].items()
                 if key.startswith("diff_model.")}
        actual_shape = tuple(state["model.scene_type_embedder.embedding_table.weight"].shape)
        expected_shape = (2 * int(cfg.dataset.num_map_ids) + int(cfg.model.label_dropout > 0),
                          int(cfg.model.hidden_dim))
        if actual_shape != expected_shape:
            raise ValueError("Incompatible scene label table")
        ldm = LDM(cfg)
        ldm.load_state_dict(state, strict=True)
        ema = ExponentialMovingAverage(ldm.parameters(), decay=cfg.train.ema_decay)
        ema.load_state_dict(ckpt["ema_state_dict"])
        ema.copy_to()
        ldm.to(args.device).eval().requires_grad_(False)
        del state, ckpt, ema
        gc.collect()
        with np.load(baseline / "regular_latents_and_conditions.npz", allow_pickle=False) as saved:
            arrays = {key: saved[key].copy() for key in saved.files}
        source_path = Path(metadata["previous_output_dir"]) / "regular_latents_and_conditions.npz"
        if sha256(source_path) != metadata["ae_latent_source_sha256"]:
            raise ValueError("Original AE latent evidence changed")
        with np.load(source_path, allow_pickle=False) as original:
            if set(original.files) != set(arrays) or not all(
                    np.array_equal(original[key], arrays[key]) for key in original.files):
                raise ValueError("Resume arrays differ from original encoded evidence")
        if (arrays["signal_point_mask"].shape != (12, 24, 20)
                or int(arrays["signal_point_mask"].sum()) != 53
                or int(arrays["signal_active"].sum()) != 6
                or not np.all(arrays["signal_batch_index"] == 0)):
            raise ValueError("Expected the six regular signals and 53 retained points")
        reference = torch.load(baseline / "ldm_forward_cpu.pt", map_location="cpu")
        if reference["data"].num_graphs != 1 or reference["data"]["lg_type"].tolist() != [0]:
            raise ValueError("Probe supports only this single regular sample")
        stats_path = Path(metadata['latent_stats']['path'])
        if sha256(stats_path) != metadata['latent_stats']['sha256']:
            raise ValueError('Latent statistics evidence changed')
        stats = pickle.loads(stats_path.read_bytes())
        for key, value in metadata['latent_stats']['values'].items():
            if not np.allclose(np.asarray(stats[key]), np.asarray(value), rtol=1e-6, atol=1e-8):
                raise ValueError('Recorded latent statistics differ from evidence')
        validate_reference(reference, arrays, stats, cfg)
        report['reference_latent_routing_verified'] = True
        condition = (
            torch.from_numpy(arrays["signal_features"]).float(),
            torch.from_numpy(arrays["signal_point_mask"]).bool(),
            torch.from_numpy(arrays["road_points"]).float(),
            torch.from_numpy(arrays["signal_active"]).bool()
        )
        report.update({"git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=code, text=True).strip(),
            "scenario_id": metadata["scenario_id"], "time_index": 10,
            "ldm_checkpoint": metadata["ldm_checkpoint"], "ldm_weight_view": "checkpoint EMA",
            "condition_npz_sha256": sha256(baseline / "regular_latents_and_conditions.npz"),
            "reference_forward_sha256": sha256(baseline / "ldm_forward_cpu.pt"),
            "torch_version": torch.__version__, "active_signal_count": 6,
            "valid_point_count": 53, "verification_scope": "one-scene signal interface and learning probe"})
        if args.device == "cuda":
            torch.cuda.reset_peak_memory_stats()
            report["gpu"] = torch.cuda.get_device_name()
        _, artifacts = run_learning_probe(
            ldm, reference, condition, args.steps, args.learning_rate, args.seed, progress=report)
        torch.save(artifacts, out / "adapter_probe.pt")
        if args.device == "cuda":
            report["peak_gpu_allocated_mib"] = torch.cuda.max_memory_allocated() / 1024**2
        report["status"] = ("signal_adapter_probe_verified" if report["condition_response_verified"]
                            else "signal_adapter_probe_no_state_response")
    except Exception:
        report["error_traceback"] = traceback.format_exc()
        print(report["error_traceback"])
    (out / "signal_adapter_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return 0 if report["status"] == "signal_adapter_probe_verified" else 1


if __name__ == "__main__":
    raise SystemExit(main())
