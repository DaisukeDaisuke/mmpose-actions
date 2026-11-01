"""Full-body drift metrics. Network 'head' always predicts every joint."""
import numpy as np

PARTS = {'all': list(range(133)), 'body': list(range(17)), 'body-feet': list(range(23)),
         'hips': [11, 12], 'knees': [13, 14], 'ankles': [15, 16], 'feet': list(range(17, 23)),
         'face': list(range(23, 91)), 'hands': list(range(91, 133))}


def drift(points, scores, reference, reference_scores):
    if points.shape != (1, 133, 3) or scores.shape != (1, 133):
        raise ValueError('Every candidate must preserve all 133 joints, including ankles and six foot points')
    result = {}
    for name, ids in PARTS.items():
        valid = reference_scores[:, ids] > .3
        error = points[:, ids] - reference[:, ids]
        xy = np.linalg.norm(error[..., :2], axis=-1)[valid]
        z = np.abs(error[..., 2])[valid]
        result[name] = {'joints': int(valid.sum()), 'xyP95_px': float(np.quantile(xy, .95)) if xy.size else None,
                       'zP95_m': float(np.quantile(z, .95)) if z.size else None,
                       'xyMax_px': float(np.max(xy)) if xy.size else None, 'zMax_m': float(np.max(z)) if z.size else None,
                       'confidenceLossCount': int((valid & (scores[:, ids] <= .3)).sum())}
    return result


def passes_fullbody_gate(comparisons):
    for name in ['body-feet', 'hips', 'knees', 'ankles', 'feet']:
        observed = [c['parts'][name] for c in comparisons if c['parts'][name]['joints']]
        if not observed:
            return False
        for part in observed:
            if part['xyP95_px'] > 3 or part['zP95_m'] > .05 or part['confidenceLossCount']:
                return False
    return True
