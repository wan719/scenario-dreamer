"""Single active signal interventions; fixed geometry and other signal states."""
import torch


def single_signal_variants(condition, signals):
    features, mask, roads, active = condition
    if features.ndim != 2 or features.shape[1] != 11 or len(signals) != len(features):
        raise ValueError('Signal metadata/feature shape differs')
    if active.dtype != torch.bool or active.shape != (len(features),) or not active.any():
        raise ValueError('Expected at least one active signal')
    if not torch.isfinite(features).all() or not torch.all((features[:, :9] == 0) | (features[:, :9] == 1)):
        raise ValueError('Expected finite one-hot signal features')
    if not torch.all(features[:, :9].sum(1) == 1):
        raise ValueError('Signal state is not one-hot')
    states = features[:, :9].argmax(1)
    if not torch.all((states[active] == 1) | (states[active] == 4)):
        raise ValueError('Active interventions require STOP or ARROW_STOP')
    variants = [('stop', condition), ('stop_repeat', condition)]
    ids = set()
    for index in active.nonzero().flatten().tolist():
        lane = int(signals[index]['raw_lane_id'])
        if lane in ids: raise ValueError('Duplicate active signal Lane ID')
        ids.add(lane)
        changed = features.clone()
        changed[index, :9] = 0
        changed[index, 6] = 1
        modified = (changed[:, :9] != features[:, :9]).any(1).nonzero().flatten().tolist()
        if modified != [index]: raise AssertionError('Expected exactly one state intervention')
        variants.append(('go_' + str(lane), (changed, mask, roads, active)))
    return variants
