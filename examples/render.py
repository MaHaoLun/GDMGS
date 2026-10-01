"""Integration function for an already-loaded, frozen ProxyGS model.

Inputs come from the caller's model/dataset loader. No checkpoint or camera
paths are embedded. Eligibility callbacks must implement the same LoD rule.
"""
import numpy as np

from gdmgs.index import AnchorIndex
from gdmgs.occluders import OccluderIndex
from gdmgs.occupancy import make_grid
from gdmgs.selection import CPUSelector
from gdmgs.tensor_selection import TensorSelector
from gdmgs.materialization import GroupMaterializer
from gdmgs.schedule import Schedule
from gdmgs.adapters.proxygs import ProxyGSDecoder
from gdmgs.adapters.render import GSplatRenderer


def render_trajectory(model, cameras, bounds, vertices, triangles, training_eyes,
                      cpu_eligibility, gpu_eligibility, cpu_cost, gpu_cost,
                      background=(0., 0., 0.)):
    """Bounds must conservatively cover support and preserve original PLY IDs."""
    points = np.concatenate((bounds[:, :3], bounds[:, 3:], vertices, training_eyes))
    lo, hi = points.min(0), points.max(0)
    margin = max(float((hi-lo).max()) * .05, 1e-6)
    domain = (lo-margin, hi+margin)
    grid = make_grid(vertices, triangles, training_eyes, level=6, domain=domain)
    anchors = AnchorIndex.build(bounds, domain=domain)
    occluders = OccluderIndex(*domain, grid['level'], grid['full_nodes'])
    cpu = CPUSelector(anchors, occluders, cpu_eligibility)
    gpu = TensorSelector(anchors, occluders, 'cuda', gpu_eligibility)
    model.eval()
    materializer = GroupMaterializer(ProxyGSDecoder(model))
    return Schedule().run(cameras, cpu, gpu, materializer, GSplatRenderer(background),
                          cpu_cost=cpu_cost, gpu_cost=gpu_cost, device='cuda')
