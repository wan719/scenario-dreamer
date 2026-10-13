"""Official AE decoding, physical-unit comparisons, and dependency-free SVG views."""
import math
from html import escape

import torch

from utils.data_helpers import unnormalize_latents, unnormalize_scene

SCALE_KEYS = ('fov', 'min_speed', 'max_speed', 'min_length', 'max_length',
              'min_width', 'max_width', 'min_lane_x', 'max_lane_x', 'min_lane_y', 'max_lane_y')


@torch.no_grad()
def decode_scene(ae, latent_pair, data, stats, scale):
    ae.eval().requires_grad_(False)
    device = next(ae.parameters()).device
    latent_pair = tuple(t.to(device) for t in latent_pair)
    values = [torch.as_tensor(stats[k], device=device, dtype=latent_pair[0].dtype)
              for k in ('agent_latents_mean','agent_latents_std','lane_latents_mean','lane_latents_std')]
    agents, lanes = unnormalize_latents(*latent_pair, *values)
    raw = ae.forward_decoder(agents, lanes, data.clone().to(device))
    if not all(torch.isfinite(v).all() for v in raw if v is not None):
        raise AssertionError('Non-finite AE decoder output')
    physical_agents, physical_roads = unnormalize_scene(
        raw[0].clone(), raw[1].clone(), **{key: float(scale[key]) for key in SCALE_KEYS})
    if not torch.isfinite(physical_agents).all() or not torch.isfinite(physical_roads).all():
        raise AssertionError('Non-finite physical scene')
    return {'agent_states': physical_agents.cpu(), 'road_points': physical_roads.cpu(),
            'normalized_agent_states': raw[0].cpu(), 'normalized_road_points': raw[1].cpu(),
            'agent_types': raw[2].cpu(), 'lane_types': None if raw[3] is None else raw[3].cpu(),
            'lane_connections': raw[4].cpu()}


def compare_scenes(first, second):
    a, b = first['agent_states'], second['agent_states']
    if a.shape != b.shape or a.ndim != 2 or a.shape[1] != 7:
        raise ValueError('Expected equal [agents,7] state arrays')
    if not torch.isfinite(a).all() or not torch.isfinite(b).all():
        raise ValueError('Non-finite comparison state')
    valid = (a[:,3:5].norm(dim=1) > 1e-6) & (b[:,3:5].norm(dim=1) > 1e-6)
    angle_a, angle_b = torch.atan2(a[:,4],a[:,3]), torch.atan2(b[:,4],b[:,3])
    delta = angle_b - angle_a
    wrapped = torch.atan2(torch.sin(delta), torch.cos(delta)) * (180 / math.pi)
    position = (a[:,:2]-b[:,:2]).norm(dim=1)
    speed = b[:,2]-a[:,2]
    rows = [{'agent_slot':i, 'position_delta_m':float(position[i]),
             'speed_delta_mps':float(speed[i]), 'heading_delta_deg':float(wrapped[i]) if valid[i] else None}
            for i in range(len(a))]
    roads = (first['road_points']-second['road_points']).norm(dim=-1)
    result = {'agent_pairing':'latent slots, not real-world track identities',
              'position_delta_mean_m':float(position.mean()), 'position_delta_max_m':float(position.max()),
              'speed_delta_mean_abs_mps':float(speed.abs().mean()), 'speed_delta_max_abs_mps':float(speed.abs().max()),
              'heading_valid_slots':int(valid.sum()),
              'heading_delta_mean_abs_deg':float(wrapped[valid].abs().mean()) if valid.any() else None,
              'road_point_delta_mean_m':float(roads.mean()), 'road_point_delta_max_m':float(roads.max())}
    if 'agent_types' in first and 'agent_types' in second:
        result['agent_type_changed_slots'] = int((first['agent_types'].argmax(-1) != second['agent_types'].argmax(-1)).sum())
    return result, rows


def scene_diagnostics(scene, reference_roads, fov):
    a, r = scene['agent_states'], scene['road_points']
    distances = (r-reference_roads).norm(dim=-1)
    return {'nonpositive_length_or_width_slots':int((a[:,5:7]<=0).any(-1).sum()),
            'near_zero_heading_vector_slots':int((a[:,3:5].norm(dim=-1)<=1e-6).sum()),
            'decoded_road_outside_fov_point_fraction':float((r.abs()>float(fov)/2).any(-1).float().mean()),
            'decoded_road_vs_input_mean_m':float(distances.mean()),
            'decoded_road_vs_input_max_m':float(distances.max())}


def _panel(scene, title, condition, scale, go=False):
    # Geometry is drawn in ego XY metres; no plotting package is required on the server.
    fov = float(scale['fov'])
    def point(x,y):
        return 24+(float(x)/fov+0.5)*320, 52+(0.5-float(y)/fov)*320
    body = ['<rect x="24" y="52" width="320" height="320" fill="#fafafa" stroke="#94a3b8"/>',
            f'<text x="184" y="24" text-anchor="middle" font-size="15">{escape(title)}</text>']
    for value in (-fov/2,0,fov/2):
        x,y = point(value,value)
        body.append(f'<text x="{x:.2f}" y="388" text-anchor="middle" font-size="10">{value:g}</text>')
        body.append(f'<text x="20" y="{y:.2f}" text-anchor="end" font-size="10">{value:g}</text>')
    body.append('<text x="184" y="408" text-anchor="middle" font-size="11">ego x (m); ego y upward</text>')
    body.append('<defs><clipPath id="'+escape(title,quote=True)+'"><rect x="24" y="52" width="320" height="320"/></clipPath></defs>')
    body.append('<g clip-path="url(#'+escape(title,quote=True)+')">')
    for lane in scene['road_points']:
        coordinates = ' '.join(f'{x:.2f},{y:.2f}' for x,y in (point(*xy) for xy in lane))
        body.append(f'<polyline points="{coordinates}" fill="none" stroke="#94a3b8" stroke-width="1.2"/>')
    features, _, _, active = condition
    for stop in features[active,9:11]:
        x,y = point(*(stop*fov/2))
        color = '#16a34a' if go else '#dc2626'
        body.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="3" fill="{color}"><title>Stop point; not a stop line</title></circle>')
    palette = ('#2563eb','#9333ea','#d97706')
    for slot,state in enumerate(scene['agent_states']):
        x,y,speed,cos,sin,length,width = [float(v) for v in state]
        norm = math.hypot(cos,sin)
        cx,cy = point(x,y)
        cls = int(scene['agent_types'][slot].argmax())
        color = palette[min(cls,len(palette)-1)]
        if length<=0 or width<=0 or norm<=1e-6:
            body.append(f'<text x="{cx:.2f}" y="{cy:.2f}" fill="#dc2626" font-size="12">×</text>')
            continue
        cos,sin = cos/norm,sin/norm
        corners = [(x+dx*cos-dy*sin,y+dx*sin+dy*cos)
                   for dx,dy in [(length/2,width/2),(length/2,-width/2),(-length/2,-width/2),(-length/2,width/2)]]
        coordinates=' '.join(f'{px:.2f},{py:.2f}' for px,py in (point(*p) for p in corners))
        body.append(f'<polygon points="{coordinates}" fill="{color}" fill-opacity="0.3" stroke="{color}"><title>slot {slot}, speed {speed:.3f} m/s</title></polygon>')
        hx,hy = point(x+length/2*cos,y+length/2*sin)
        body.append(f'<line x1="{cx:.2f}" y1="{cy:.2f}" x2="{hx:.2f}" y2="{hy:.2f}" stroke="{color}" stroke-width="2"/>')
        body.append(f'<text x="{cx:.2f}" y="{cy-3:.2f}" font-size="8">{slot}</text>')
    body.append('</g>')
    body.append('<text x="184" y="426" text-anchor="middle" font-size="10">Blue: vehicle; purple: pedestrian; orange: cyclist</text>')
    return '\n'.join(body)


def write_scene_views(directory, scenes, condition, scale):
    paths=[]
    labels={'baseline':'Baseline','stop':'Adapter_STOP','go':'Adapter_synthetic_GO'}
    panels=[]
    for index,(key,title) in enumerate(labels.items()):
        panel=_panel(scenes[key],title,condition,scale,go=key=='go')
        svg='<svg xmlns="http://www.w3.org/2000/svg" width="368" height="430" viewBox="0 0 368 430"><rect width="368" height="430" fill="white"/>'+panel+'</svg>'
        filename=key+'.svg'; (directory/filename).write_text(svg); paths.append(filename)
        panels.append(f'<g transform="translate({index*368},0)">{panel}</g>')
    filename='comparison.svg'
    (directory/filename).write_text('<svg xmlns="http://www.w3.org/2000/svg" width="1104" height="465" viewBox="0 0 1104 465"><rect width="1104" height="465" fill="white"/>'+''.join(panels)+'<text x="552" y="448" text-anchor="middle" font-size="11">Single generated scene; slot labels are not track IDs. Stop point markers are not stop lines.</text></svg>')
    return paths+[filename]
