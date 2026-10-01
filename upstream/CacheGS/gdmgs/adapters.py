"""Fresh explicit-ID inference; never calls the legacy selection/cache pipeline."""

from dataclasses import dataclass, field

import torch

from .runtime.camera_key import CameraQueryKey, appearance_key


def _tensor_stamp(value):
    if value is None:
        return None
    return id(value), value._version, tuple(value.shape), value.dtype, value.device


def _pose_stamp(state):
    return tuple(_tensor_stamp(getattr(state, name, None))
                 for name in ("anchor_mask", "prog_ratio", "transition_mask"))


def _materialized_stamp(value):
    batch = value.batch
    metadata = getattr(batch, "bundle_metadata", None)
    fields = [value.anchor_ids]
    fields.extend(getattr(batch, name, None)
                  for name in ("xyz", "color", "opacity", "scaling", "rotation", "selection_mask"))
    fields.extend(getattr(metadata, name, None) for name in (
        "request_anchor_ids", "request_level_ids", "row_owner_ids", "row_owner_levels",
        "row_offset_slots", "counts", "offsets"))
    return (getattr(batch, "sh_degree", None), tuple(_tensor_stamp(item) for item in fields))


@dataclass(frozen=True)
class PreparedSelection:
    anchor_ids: torch.Tensor
    pose_state: object
    camera_key: CameraQueryKey
    scene_token: str
    _state_stamp: tuple = field(init=False, repr=False)

    def __post_init__(self):
        object.__setattr__(self, "_state_stamp", _pose_stamp(self.pose_state))

    def validate(self):
        if _pose_stamp(self.pose_state) != self._state_stamp:
            raise RuntimeError("Prepared pose state changed after the FoV query.")


@dataclass(frozen=True)
class MaterializedSelection:
    batch: object
    anchor_ids: torch.Tensor
    camera_key: CameraQueryKey
    scene_token: str
    _payload_stamp: tuple = field(init=False, repr=False)

    def __post_init__(self):
        object.__setattr__(self, "_payload_stamp", _materialized_stamp(self))

    def validate(self):
        if _materialized_stamp(self) != self._payload_stamp:
            raise RuntimeError("Materialized tensors changed after decoding.")


def prepare_selection(session, camera, pipe, bg_color, ape_code=-1):
    from gaussian_renderer.visibility import sample_visibility_pose_local

    session.validate()
    key = CameraQueryKey.from_camera(camera, session.scene.iteration, ape_code)
    with torch.no_grad():
        state = session.model.compute_pose_local_state(camera, session.scene.iteration)
        sample = sample_visibility_pose_local(camera, session.model, pipe, bg_color,
                                              pose_state=state, return_mask=False)
        ids = session.scene.validate_ids(sample.indices)
    return PreparedSelection(ids, state, key, session.scene.token)


def materialize_selected(session, selected_ids, camera, *, ape_code=-1, prepared=None):
    session.validate()
    ids = session.scene.validate_ids(selected_ids)
    key = CameraQueryKey.from_camera(camera, session.scene.iteration, ape_code)
    if prepared is not None:
        if prepared.scene_token != session.scene.token or prepared.camera_key != key:
            raise ValueError("Prepared FoV belongs to another camera or scene session.")
        prepared.validate()
        state = prepared.pose_state
    else:
        with torch.no_grad():
            state = session.model.compute_pose_local_state(camera, session.scene.iteration)
    ape = appearance_key(ape_code)
    with torch.no_grad():
        batch = session.model.generate_neural_gaussians(
            camera, ids, ape, build_descriptor=False,
            pose_state=state, return_bundle_metadata=True,
        )
    metadata = getattr(batch, "bundle_metadata", None)
    if metadata is None or not torch.equal(metadata.request_anchor_ids, ids):
        raise RuntimeError("Decoder did not preserve the explicit anchor request order.")
    session.validate()
    return MaterializedSelection(batch, ids, key, session.scene.token)


def rasterize_materialized(session, materialized, camera, bg_color, render_mode="RGB", *, ape_code=-1):
    from gaussian_renderer.render import rasterize_batch

    session.validate()
    key = CameraQueryKey.from_camera(camera, session.scene.iteration, ape_code)
    if materialized.scene_token != session.scene.token or materialized.camera_key != key:
        raise ValueError("Fresh materialization belongs to another camera or scene session.")
    materialized.validate()
    ids = session.scene.validate_ids(materialized.anchor_ids)
    metadata = getattr(materialized.batch, "bundle_metadata", None)
    if metadata is None or not torch.equal(metadata.request_anchor_ids, ids):
        raise ValueError("Materialized request IDs no longer match the decoder ownership.")
    metadata.validate(row_count=materialized.batch.xyz.shape[0])
    visible = torch.zeros(session.scene.anchor_count, dtype=torch.bool, device=ids.device)
    visible[ids] = True
    with torch.no_grad():
        result = rasterize_batch(camera, session.model, materialized.batch, bg_color, render_mode,
                                 visible_mask=visible, visibility_summary={"total": int(ids.numel())})
    result.update(selected_anchor_ids=ids, bundle_metadata=materialized.batch.bundle_metadata,
                  camera_key=key, scene_token=session.scene.token, pipeline="gdmgs-fresh")
    return result
