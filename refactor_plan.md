# Rollout PPO Refactor Plan

Sanity check command (run after meaningful milestones):
```
python ./bin/Rollout.py PpoCus RealXarm7DualDemo \
  --wait_before_start \
  --skip_draw 50000 \
  --save_rollout \
  --config ./envs/configs/RealXarm7DualDemoEnv.yaml \
  --checkpoint ./checkpoint/PpoCus/DualBoxRotationAblated/ckpt_26.pt
```

## Workstream 1: Marker subsystem isolation
**Goal:** Move marker configuration, detection, caching, and lifecycle management out of `RolloutPpoCus` into a dedicated component under `marker_detection.py`.

### Tasks
- [ ] Create `MarkerManager` abstraction (config parsing, calibration loading, worker lifecycle, frame submission, transform cache).
- [ ] Update `marker_detection.py` to host `MarkerManager` and keep `FrontCameraDetectionWorker` internal.
- [ ] Replace marker-related fields/methods in `RolloutPpoCus` with `self.marker_manager` usage.
- [ ] Ensure marker-optional paths degrade gracefully when calibration or cameras are missing.
- [ ] Document the new marker subsystem usage (README / inline docstrings).
- [ ] Verify with the sanity command when marker config is present and when it is absent.

## Workstream 2: State gathering refactor
**Goal:** Make `get_state` composable and side-effect-light so sensor handling, buffer updates, and marker refreshes are easy to reason about.

### Tasks
- [ ] Extract helper(s) for assembling normalized state vectors (`_collect_state_components`).
- [ ] Extract buffer management into `_update_state_buffers` and image equivalents.
- [ ] Route marker cache refreshes through `MarkerManager`; remove `_refresh_marker_cache`.
- [ ] Leave `get_state` responsible only for orchestrating helpers and returning the tensor.
- [ ] Add unit-level coverage for the new helpers if feasible; otherwise rely on targeted assertions within Rollout.
- [ ] Re-run the sanity command to confirm no regression in observation shapes.

## Workstream 3: Policy inference refactor
**Goal:** Break down `infer_policy` into small, testable pieces that separate profiling, model inference, action scaling, and logging.

### Tasks
- [ ] Introduce helper methods (`_maybe_profile`, `_run_policy_forward`, `_scale_and_clip_action`, `_log_policy_outputs`, `_queue_policy_action`).
- [ ] Move CSV/log initialization logic into a dedicated `_init_debug_logging` called from `setup_policy`.
- [ ] Keep gripper conversion and joint limits encapsulated inside helpers with minimal side effects.
- [ ] Centralize `policy_action_list` updates so batching semantics are obvious.
- [ ] Validate by running the sanity command and checking action logs for shape consistency.

## Workstream 4: Logging & profiling consolidation
**Goal:** Standardize how rollout diagnostics are configured and flushed so instrumentation does not clutter functional code.

### Tasks
- [ ] Create a lightweight logging utility or structured dict to hold log paths, writers, and headers.
- [ ] Relocate profiling hook installation and teardown into `_init_profiling` / `_finalize_profiling` helpers.
- [ ] Ensure all logging/profiling state lives in clearly named attributes (e.g., `self.debug_logging`, `self.profile_data`).
- [ ] Update `print_statistics` to consume the new structures.
- [ ] Sanity command should run with profiling enabled and disabled to ensure hooks behave identically.

## Workstream 5: Model definition extraction
**Goal:** Share a single `ManiSkillPpoAgent` implementation across rollouts to simplify maintenance.

### Tasks
- [ ] Create `robo_manip_baselines/policy/ppo_cus/models.py` (or shared module) containing `_layer_init` + `ManiSkillPpoAgent`.
- [ ] Update `RolloutPpoCus`, `RolloutPpoCus_Hinansaseta`, and any similar files to import the shared model.
- [ ] Confirm checkpoints still load without key mismatches.
- [ ] Update documentation to reflect the new module.
- [ ] Run the sanity command for at least one policy using the shared model.

## Workstream 6: Marker docs & tooling
**Goal:** Provide clear instructions for calibration assets, marker IDs, and troubleshooting.

### Tasks
- [ ] Expand `README_Sim2Real_RL.md` (or a new doc) with MarkerManager usage, calibration file expectations, and debugging steps.
- [ ] Provide example config snippets for defining markers in `model_meta_info`.
- [ ] Add troubleshooting tips (e.g., camera not found, missing IDs) linked to log messages.
- [ ] Verify documentation steps manually using the sanity command sequence.

---

Progress is tracked by moving completed tasks to `done_refactor_plan.md` (create/update as needed).
