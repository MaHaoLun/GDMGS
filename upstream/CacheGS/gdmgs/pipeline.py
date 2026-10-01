"""Stage-B cache-off pipeline. Mesh queries, bundle caching and scheduling follow later."""

import os

from .adapters import prepare_selection, materialize_selected, rasterize_materialized
from .runtime.session import InferenceSession


def validate_environment():
    cache = os.environ.get("CACHE_ENABLE", "0").strip()
    if cache not in ("", "0") or os.environ.get("PRECOMP_INDICES_PATH", "").strip():
        raise ValueError("GDM-GS fresh mode requires CACHE_ENABLE=0 and no PRECOMP_INDICES_PATH.")


class FreshPipeline:
    def __init__(self, model, checkpoint_path, iteration):
        validate_environment()
        self.session = InferenceSession(model, checkpoint_path, iteration)

    def render(self, camera, pipe, bg_color, render_mode="RGB", ape_code=-1, *, selected_ids=None):
        validate_environment()
        prepared = None
        if selected_ids is None:
            prepared = prepare_selection(self.session, camera, pipe, bg_color, ape_code)
            selected_ids = prepared.anchor_ids
        materialized = materialize_selected(self.session, selected_ids, camera,
                                            ape_code=ape_code, prepared=prepared)
        return rasterize_materialized(self.session, materialized, camera, bg_color, render_mode,
                                       ape_code=ape_code)

    def close(self):
        self.session.close()

    def __enter__(self):
        self.session.validate()
        return self

    def __exit__(self, *exc):
        self.close()
