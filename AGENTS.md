# Repository working conventions

## Dataset-generation annotation provenance gate

- The user fixed the source for all subsequent rollout, validation, and dataset
  generation on 2026-08-31: use scripted/geometry ground truth. Do not ask again
  before each launch. Always pass `--annotation-source scripted` explicitly and
  retain the top-level pickle `annotation_source=scripted`.
- Never relabel VLM predictions as ground truth. Policy-facing skill and guidance
  targets must come from the task's scripted/geometry annotator and must be
  checked against the same-frame 3-D geometry and camera calibration.
- For FurnitureBench collection through `src/eval/evaluate_model.py`, include
  `--annotate-skill --enable-annotation-verify --annotation-source scripted`.
  ManiSkill and AutoMate must obey the same provenance contract before their new
  outputs can enter this campaign.
- Before launch, record the exact command and resolved annotation source in
  tmux/run metadata. Fail closed if the requested source is not `scripted` or if
  generated pickle provenance and geometry validation do not agree.
- FurnitureBench collection must preserve raw wrist/front RGB-D pixels. During
  collection record `skill`, guidance geometry, `guidance_point_2d`, and camera
  calibration, but never pass `--guidance-point-on-image`,
  `--grasp-annotation-on-image`, `--grasp-part-annotate`, or `--skill-on-image`
  for saved pickles. Render markers only in pickle-to-LMDB conversion through
  an explicitly confirmed `--image-annotation-mode`; keep that mode in LMDB
  metadata and never modify the source pickle.
- Prefer the maintained two-stage entry points under
  `scripts/data_collection/` for FurnitureBench campaigns. Use a new raw
  `--output-suffix` for every campaign; do not mix newly collected and existing
  pickle files unless the user explicitly approves it.
- After collection, audit every output. `annotation_source` must equal
  `scripted`, VLM metadata must be absent, and `guidance_point_2d` must be the
  calibrated projection of the same-frame GT 3-D guidance point. Audit bounds,
  nulls, skill/FSM state, and recorded-vs-reprojected agreement in addition to
  pickle schema.
- Historical pilots remain diagnostic evidence only. Do not reuse their pickle
  files in the new production campaign.

## Working-tree hygiene

- Do not add one-off, exploratory, debugging, inspection, or data-migration scripts to `scripts/`.
- Put temporary scripts and their generated artifacts under a task-specific directory in `logs/`, for example `logs/<task-name>/tools/`. The `logs/` tree is ignored by Git and temporary files there should be removed when they are no longer useful.
- Reserve `scripts/` for reusable project entry points that are expected to remain maintained. A new reusable script should have a stable CLI, documentation or a clear caller, and focused tests where practical.
- Do not add generated manifests, raw diagnostic dumps, copied runtime state, temporary samples, or command transcripts to tracked source/report directories. Keep them in `logs/`; only concise, intentionally maintained reports and small durable reference data belong under `reports/`.
- Before creating a helper script, prefer an existing project command or a short one-time command when that is sufficient.
- Preserve unrelated dirty-worktree changes. Do not reset, clean, stash, overwrite, or reformat files outside the active task.

## Configured experiment environments

- Reuse environments that the user has already confirmed are configured. Do
  not create a new Conda environment for each rollout, diagnostic, camera
  adjustment, or data-collection run.
- On `r218` (`lht-3060-12G`), run ManiSkill tools with
  `/home/hy/anaconda3/envs/rr-maniskill/bin/python` and the editable source at
  `/data/hy/ManiSkill`. Do not create task-specific replacements such as a
  separate camera environment unless the user explicitly requests one.
- Keep host environments independent. A local r218 path must not be assumed to
  exist on a 4090 server; use the already configured environment for that host,
  and keep server Conda environments under the NAS Conda installation.
- Create or rebuild an environment only if it is missing or demonstrably
  broken, or if the user explicitly asks for a clean rebuild. Before doing so,
  state why the configured environment cannot satisfy the task.
