## Marker handling refactor plan (RolloutPpoCus → marker_detection)

Goal: move marker detection/config/cache responsibilities out of `RolloutPpoCus` into `marker_detection.py` to shrink rollout complexity and isolate AprilTag deps.

### Current responsibilities inside RolloutPpoCus
- Parse marker meta info (`marker_definitions`, `required_marker_ids`, `marker_name_map`, `marker_size_map`, `marker_camera_names`) during `setup_model_meta_info`.
- Load T_base_to_camera, camera intrinsics, and initialize `FrontCameraDetectionWorker`.
- Submit frames from `get_images`/`get_state` (`_submit_marker_frame`) and poll/pull cached transforms (`_refresh_marker_cache`, `get_latest_marker_transforms`).
- Own all marker-related fields and lifecycle (start/stop) intertwined with rollout lifecycle.

### Proposed abstraction
- Introduce `MarkerManager` in `marker_detection.py`:
  - `MarkerManager.from_meta(meta_info, cameras, default_tag_size=...)` to parse marker config, load calibration (T_base_to_camera), extract intrinsics.
  - `start()`/`stop()` to manage background worker.
  - `submit_frame(camera_name, rgb_image)` to feed frames.
  - `get_transforms(required_ids=None, poll=False)` to fetch latest transforms/timestamps; maintain cache of last-known transforms.
  - Access to marker metadata (id→name/size mapping, active camera).
- Keep `FrontCameraDetectionWorker` internal to `MarkerManager`.

### RolloutPpoCus changes (high level)
- Replace marker-related fields with a single `self.marker_manager`.
- In `setup_model_meta_info`, call `MarkerManager.from_meta(...)` if meta has marker config.
- In `reset`/`close`, call `marker_manager.start()/stop()` appropriately.
- In `get_state`/`get_images`, replace `_submit_marker_frame` and cache polling with `marker_manager.submit_frame`/`marker_manager.get_transforms`.
- Remove `_refresh_marker_cache`, `_marker_worker`, `_marker_camera_active`, related marker fields from RolloutPpoCus.

### Notes
- Keep marker functions optional; if calibration or deps are missing, manager should gracefully disable detection and return empty dict.
- Ensure thread safety and minimal blocking (same semantics as current worker).
- Update `marker_detection.py` tests/usages accordingly.

---

## Policy model extraction plan (ManiSkillPpoAgent)

Goal: move `ManiSkillPpoAgent` out of rollout files into a shared module for readability and reuse.

### Current state
- `ManiSkillPpoAgent` is defined inline in `policy/ppo_cus/RolloutPpoCus.py`, `RolloutPpoCus_Hinansaseta.py`, and `policy/ppo/RolloutPpo.py` with small variations.
- `_layer_init` utility is also embedded alongside.

### Proposed steps
1) Consolidate the canonical agent definition (decide which variant to keep, or parameterize differences).
2) Create `policy/ppo_cus/models.py` (or a shared `policy/ppo_common/models.py`) containing:
   - `_layer_init`
   - `ManiSkillPpoAgent` (matching checkpoint structure)
3) Update rollouts to `from .models import ManiSkillPpoAgent` (or shared path) and remove inline class/utility.
4) Ensure checkpoint compatibility: class name and layer shapes must remain unchanged so existing `state_dict` loads.
5) Adjust any tests/imports accordingly.

### Notes
- If Hinansaseta variant differs (e.g., architecture hyperparams), consider either keeping a single canonical definition or exposing constructor args for differences.
