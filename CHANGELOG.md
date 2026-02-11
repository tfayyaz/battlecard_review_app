# Changelog

## 2026-02-11

### Branch Targets
- GitHub branch: `feat/dev-runlocal-autoscaling-branch` (HEAD `4a3eb31`)
- Upstream: `origin/feat/dev-runlocal-autoscaling-branch`
- Postgres branch target (dev autoscaling): `projects/battlecards-review-dev/branches/dev`
- Postgres endpoint target (rw): `projects/battlecards-review-dev/branches/dev/endpoints/rw`

### Added
- Identity and attribution model:
  - New `users`, `roles`, `user_assigned_roles`, `agents`, `content_copy_log` tables.
  - New identity helper module: `identity.py`.
  - Default identity/agent/role seeding on init.
- Entity attribution columns on workflow and artifact tables:
  - `created_by_user_id`, `created_by_agent_id`, `updated_by_user_id`, `updated_by_agent_id`.
- Copy lineage columns for regeneration/reuse:
  - `source_generation_id`, `source_session_id`,
  - `copied_from_generation_id`, `copied_from_key_diff_id`.
- New attribution API:
  - `GET /api/battlecard/<battlecard_id>/attribution`.
- Local dev tooling for autoscaling Postgres:
  - `scripts/run_local_autoscaling_dev.sh`
  - `scripts/migrate_provisioned_to_autoscaling.sh`

### Changed
- Workflow creation now stamps actor metadata and logs reuse lineage when previous generation context is copied.
- Step 3 category save now writes actor metadata into:
  - `session_category_selections`
  - `workflow_artifacts`
  - `agent_turns`
- Runner write paths now stamp step-specific agent attribution:
  - Step 4 (`custom_battlecard_agent_pass_step_4`)
  - Step 5 (`custom_battlecard_agent_pass_step_5`)
  - Step 6 fact-check (`custom_battlecard_agent_pass_step_6`)
- Battlecard/review UI now exposes attribution metadata and copy lineage.

### Fixed
- Fixed crash in Step 4 save path:
  - `_save_skeletons_to_lakebase` used `actor[...]` without initializing `actor`.
  - Resolution: initialize with `_ensure_actor_records(conn, 4, self.model_name)`.
- Fixed identity upsert conflict behavior:
  - `ensure_user()` now handles email/user-id mismatch by reusing existing `user_id` for the same email.
- Fixed review-save consistency work in recent commits:
  - Preserve review status when comment updates are saved.
  - Improve deterministic save behavior and debug toast visibility.

### Recent Commit Trail (same GitHub branch)
- `4a3eb31` Preserve review status when comments are saved and merge scope feedback rows
- `811d559` Make review saves deterministic and add DB-backed save debug toast
- `ab75802` Improve battlecard evidence UI with inline key-diff details and fact-check controls
- `c9bd16c` Update battlecard review UI and step 5 status handling
- `be1fe6b` Add incremental Step 4 Lakebase writes and timeline telemetry
- `ed93c5f` Add admin purge endpoint and dashboard delete action
- `32f013e` Add strict prompt-schema verification and Databricks 429 backoff
- `2547020` Add Step 4 runtime template/schema controls and strict context-source wiring
- `e5d4fdf` Use info button for key-diff details and expose DB-backed diff metadata
- `a3221f2` Add v10 granular citation workflow and battlecard review UX updates
- `78e6532` Migrate category storage to `product_category_catalog` only

### Notes
- All entries above are tied to the Git branch and Postgres branch targets listed in **Branch Targets**.
- If you switch Postgres branch/endpoint, update this file with the exact `projects/.../branches/.../endpoints/...` path used for validation.
