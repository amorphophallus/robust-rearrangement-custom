# Repository working conventions

## Working-tree hygiene

- Do not add one-off, exploratory, debugging, inspection, or data-migration scripts to `scripts/`.
- Put temporary scripts and their generated artifacts under a task-specific directory in `logs/`, for example `logs/<task-name>/tools/`. The `logs/` tree is ignored by Git and temporary files there should be removed when they are no longer useful.
- Reserve `scripts/` for reusable project entry points that are expected to remain maintained. A new reusable script should have a stable CLI, documentation or a clear caller, and focused tests where practical.
- Do not add generated manifests, raw diagnostic dumps, copied runtime state, temporary samples, or command transcripts to tracked source/report directories. Keep them in `logs/`; only concise, intentionally maintained reports and small durable reference data belong under `reports/`.
- Before creating a helper script, prefer an existing project command or a short one-time command when that is sufficient.
- Preserve unrelated dirty-worktree changes. Do not reset, clean, stash, overwrite, or reformat files outside the active task.
