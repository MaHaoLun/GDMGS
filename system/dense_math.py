"""Camera interpolation and labels, independent of the rendering runtime."""
import numpy as np

DENSITIES = (1, 2, 4)
KMAX = 8
FEATURES = ['k', 'distance_max', 'rotation_max', 'path_distance',
            'jaccard_min', 'coverage_min', 'union_ratio', 'count_ratio']


def interpolate_pose(aR, aT, bR, bT, t):
    from scipy.spatial.transform import Rotation, Slerp
    if not 0 <= t <= 1:
        raise ValueError('interpolation must stay inside the camera interval')
    # Stored R is camera-to-world, T is world-to-camera translation.
    ca, cb = -aR @ aT, -bR @ bT
    R = Slerp([0., 1.], Rotation.from_matrix(np.stack([aR, bR])))([t]).as_matrix()[0]
    center = (1-t)*ca + t*cb
    return R, -R.T @ center


def features(centers, rotations, ids, scale, start, k):
    if k < 2 or start+k > len(ids) or not np.isfinite(scale) or scale <= 0:
        raise ValueError('invalid candidate')
    source = ids[start]
    union = source
    ds, angles, js, cov = [], [], [], []
    for j in range(start+1, start+k):
        ds.append(float(np.linalg.norm(centers[j]-centers[start])/scale))
        cosine = (np.trace(rotations[start].T @ rotations[j])-1)/2
        angles.append(float(np.degrees(np.arccos(np.clip(cosine, -1, 1)))))
        common = len(np.intersect1d(source, ids[j], assume_unique=True))
        js.append(common/max(1, len(source)+len(ids[j])-common))
        cov.append(common/max(1, len(ids[j])))
        union = np.union1d(union, ids[j])
    path = sum(np.linalg.norm(centers[j]-centers[j-1])/scale
               for j in range(start+1, start+k))
    return [k, max(ds), max(angles), float(path), min(js), min(cov),
            len(union)/max(1, len(source)),
            max(len(ids[j]) for j in range(start, start+k))/max(1, len(source))]


def label(candidates, available):
    expected = set(range(2, min(KMAX, available)+1))
    if len(candidates) != len(expected) or {r['k'] for r in candidates} != expected:
        raise ValueError('missing or duplicate candidate labels')
    k = max([1] + [r['k'] for r in candidates if r['safe']])
    return dict(k_star=k, search_limit_censored=k == KMAX,
                trajectory_end_censored=k == available and available < KMAX)


def predict(tree, x):
    if tree is None or not np.isfinite(x).all():
        return 0.
    node = 0
    while tree['left'][node] != -1:
        node = tree['left'][node] if x[tree['feature'][node]] <= tree['cut'][node] else tree['right'][node]
    return tree['probability'][node]
