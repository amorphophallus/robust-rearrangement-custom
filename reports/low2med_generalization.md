# Low → Med Randomness Generalization Evaluation

**Date**: 2026-06-26
**Purpose**: Test position generalization — models trained on low randomness, evaluated on med randomness
**Eval settings**: N_ENVS=3, N_ROLLOUTS=12, observation-space=image (BC) / state (RPPO), randomness=med

---

## Experiment Schedule

### Phase 1: Multi-task BC (5 models × 3 tasks = 15 eval runs)

| # | Condition | RUN_ID | EVAL flags | Tasks |
|---|---------|--------|-----------|-------|
| 1 | rgbd+gp | autumn-dust-13 | GP=true | one_leg, round_table, lamp |
| 2 | rgbd+gp+skill | fresh-tree-11 | GP=true, SKILL=true | one_leg, round_table, lamp |
| 3 | rgbd+colored gp | absurd-voice-2 | GP=true, COLOR=true | one_leg, round_table, lamp |
| 4 | rgbd | clear-water-12 | *(none)* | one_leg, round_table, lamp |
| 5 | rgb | true-firefly-8 | *(none)* | one_leg, round_table, lamp |

> ⚠️ icy-vortex-9 (0610, labeled "rgbd+skill") skipped — actual config is GP=True, skill=False (same as autumn-dust-13).

### Phase 2: RPPO (3 models × 1 task = 3 eval runs)

| # | Task | Checkpoint | obs |
|---|------|-----------|-----|
| 1 | one_leg | `checkpoints/rppo/one_leg/low/actor_chkpt.pt` | state |
| 2 | round_table | `checkpoints/rppo/round_table/low/actor_chkpt.pt` | state |
| 3 | lamp | `checkpoints/rppo/lamp/low/actor_chkpt.pt` | state |

### Phase 3: Single-task BC (6 models × 1 task = 6 eval runs)

| # | Condition | RUN_ID | EVAL flags | Task |
|---|---------|--------|-----------|------|
| 1 | rgbd+gp | dauntless-breeze-2 | GP=true | round_table |
| 2 | rgbd+skill | misunderstood-firebrand-6 | SKILL=true | round_table |
| 3 | rgbd+gp+skill | vocal-bush-11 | GP=true, SKILL=true | round_table |
| 4 | rgbd+colored gp | gentle-fog-7 | GP=true, COLOR=true | round_table |
| 5 | rgbd | breezy-rain-3 | *(none)* | round_table |
| 6 | rgb | fiery-snowball-4 | *(none)* | round_table |

---

## Results

**Eval completed: 2026-06-27**. All models trained on low randomness, evaluated on med randomness, 12 rollouts per task.

| Type | Condition | RUN_ID | one_leg | round_table | lamp | **Overall** |
|------|---------|--------|:---:|:---:|:---:|:---:|
| mt-bc | rgbd+gp | autumn-dust-13 | 25.00% (3/12) | 0.00% (0/12) | 0.00% (0/12) | **8.33% (3/36)** |
| mt-bc | rgbd+gp+skill | fresh-tree-11 | 8.33% (1/12) | 0.00% (0/12) | 0.00% (0/12) | **2.78% (1/36)** |
| mt-bc | rgbd+colored gp | absurd-voice-2 | 16.67% (2/12) | 0.00% (0/12) | 0.00% (0/12) | **5.56% (2/36)** |
| mt-bc | rgbd | clear-water-12 | 0.00% (0/12) | 0.00% (0/12) | 0.00% (0/12) | **0.00% (0/36)** |
| mt-bc | rgb | true-firefly-8 | 0.00% (0/12) | 0.00% (0/12) | 0.00% (0/12) | **0.00% (0/36)** |
| rppo | one_leg | — | 25.00% (3/12) | — | — | **25.00% (3/12)** |
| rppo | round_table | — | — | 16.67% (2/12) | — | **16.67% (2/12)** |
| rppo | lamp | — | — | — | 8.33% (1/12) | **8.33% (1/12)** |
| st-bc | rgbd+gp | dauntless-breeze-2 | — | 8.33% (1/12) | — | **8.33% (1/12)** |
| st-bc | rgbd+skill | misunderstood-firebrand-6 | — | 0.00% (0/12) | — | **0.00% (0/12)** |
| st-bc | rgbd+gp+skill | vocal-bush-11 | — | 0.00% (0/12) | — | **0.00% (0/12)** |
| st-bc | rgbd+colored gp | gentle-fog-7 | — | 8.33% (1/12) | — | **8.33% (1/12)** |
| st-bc | rgbd | breezy-rain-3 | — | 0.00% (0/12) | — | **0.00% (0/12)** |
| st-bc | rgb | fiery-snowball-4 | — | 0.00% (0/12) | — | **0.00% (0/12)** |

### Key findings

1. 从成功率看起来 guidance point conditioned 更能做空间上的泛化
2. 然后看 failure case 的话，单任务 colored guidance point 会比单色 guidance point 的点跟随更好
3. colored gp 和 gp 主要的 failure case 是 grasp 失败或者 grasp pose OOD 导致 place 失败。

### Step-level success rates

#### MT-BC: one_leg skill success rates (cascading)

Data cross-validated from stdout logs and `logs/evaluate_model/one_leg/<checkpoint>/*.json`. Each cell = `completion_count / state_count` from JSON; percentage from stdout.

| Condition | RUN_ID | top-leg-pick | top-leg-push | leg-top-pick | leg-top-place | leg-top-insert | leg-top-screw | assembly: top-leg |
|---------|--------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| rgbd+gp | autumn-dust-13 | 75.00% (9/12) | 66.67% (6/9) | 100.00% (6/6) | 66.67% (4/6) | 100.00% (4/4) | 75.00% (3/4) | 25.00% (3/12) |
| rgbd+gp+skill | fresh-tree-11 | 83.33% (10/12) | 70.00% (7/10) | 71.43% (5/7) | 20.00% (1/5) | 100.00% (1/1) | 100.00% (1/1) | 8.33% (1/12) |
| rgbd+colored gp | absurd-voice-2 | 83.33% (10/12) | 60.00% (6/10) | 83.33% (5/6) | 60.00% (3/5) | 66.67% (2/3) | 100.00% (1/1) | 16.67% (2/12) |
| rgbd | clear-water-12 | 33.33% (4/12) | 50.00% (2/4) | 0.00% (0/2) | — | — | — | 0.00% (0/12) |
| rgb | true-firefly-8 | 8.33% (1/12) | 0.00% (0/1) | — | — | — | — | 0.00% (0/12) |

> `—` = no rollout reached this skill. Data source: `skill_state_counts` / `skill_completion_counts` in `logs/evaluate_model/one_leg/<checkpoint_name>/*.json`, cross-validated against stdout `Skill success rates`.

#### MT-BC: round_table skill success rates (cascading)

| Condition | RUN_ID | top-leg-push | leg-top-pick | leg-top-place | leg-top-insert | leg-top-screw | base-leg-pick | base-leg-place | base-leg-insert | base-leg-screw | asm: top-leg | asm: leg-base |
|---------|--------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| rgbd+gp | autumn-dust-13 | 91.67% (11/12) | 54.55% (6/11) | 50.00% (3/6) | 66.67% (2/3) | 50.00% (1/2) | 100.00% (1/1) | 0.00% (0/1) | — | — | 8.33% (1/12) | 0.00% (0/1) |
| rgbd+gp+skill | fresh-tree-11 | 83.33% (10/12) | 40.00% (4/10) | 25.00% (1/4) | 100.00% (1/1) | 0.00% (0/1) | — | — | — | — | 0.00% (0/12) | — |
| rgbd+colored gp | absurd-voice-2 | 66.67% (8/12) | 62.50% (5/8) | 0.00% (0/5) | — | — | — | — | — | — | 0.00% (0/12) | — |
| rgbd | clear-water-12 | 83.33% (10/12) | 30.00% (3/10) | 100.00% (3/3) | 100.00% (3/3) | 100.00% (2/2) | 33.33% (1/3) | 0.00% (0/1) | — | — | 25.00% (3/12) | 0.00% (0/3) |
| rgb | true-firefly-8 | 58.33% (7/12) | 28.57% (2/7) | 50.00% (1/2) | 100.00% (1/1) | 0.00% (0/1) | — | — | — | — | 0.00% (0/12) | — |

#### MT-BC: lamp skill success rates (cascading)

| Condition | RUN_ID | base-bulb-push | bulb-base-pick | bulb-base-place | bulb-base-insert | bulb-base-screw | hood-base-pick | asm: base-bulb | asm: base-hood |
|---------|--------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| rgbd+gp | autumn-dust-13 | 66.67% (8/12) | 87.50% (7/8) | 14.29% (1/7) | 100.00% (1/1) | 0.00% (0/1) | — | 0.00% (0/12) | — |
| rgbd+gp+skill | fresh-tree-11 | 58.33% (7/12) | 28.57% (2/7) | 50.00% (1/2) | 100.00% (1/1) | 0.00% (0/1) | — | 0.00% (0/12) | — |
| rgbd+colored gp | absurd-voice-2 | 50.00% (6/12) | 50.00% (3/6) | 33.33% (1/3) | 100.00% (1/1) | 100.00% (1/1) | 0.00% (0/1) | 8.33% (1/12) | 0.00% (0/1) |
| rgbd | clear-water-12 | 16.67% (2/12) | 0.00% (0/2) | — | — | — | — | 0.00% (0/12) | — |
| rgb | true-firefly-8 | 8.33% (1/12) | 0.00% (0/1) | — | — | — | — | 0.00% (0/12) | — |

> **RPPO**: State-based eval does not support `--annotate-skill`. JSON logs have empty `skill_success_rates` / `assembly_step_success_rates`. No step-level data available.

#### ST-BC: round_table assembly step rates

| Condition | RUN_ID | top-leg | leg-base |
|---------|--------|:---:|:---:|
| rgbd+gp | dauntless-breeze-2 | 16.67% (2/12) | 50.00% (1/2) |
| rgbd+skill | misunderstood-firebrand-6 | 8.33% (1/12) | 0.00% (0/1) |
| rgbd+gp+skill | vocal-bush-11 | 8.33% (1/12) | 0.00% (0/1) |
| rgbd+colored gp | gentle-fog-7 | 25.00% (3/12) | 33.33% (1/3) |
| rgbd | breezy-rain-3 | 8.33% (1/12) | 0.00% (0/1) |
| rgb | fiery-snowball-4 | 16.67% (2/12) | 0.00% (0/2) |

#### ST-BC: round_table skill-level success rates (cascading)

Data cross-validated from stdout logs and `logs/evaluate_model/round_table/*.json`. Each cell = `completion_count / state_count` from JSON; percentage = stdout.

| Condition | RUN_ID | top-leg-push | leg-top-pick | leg-top-place | leg-top-insert | leg-top-screw | base-leg-pick | base-leg-place | base-leg-insert | base-leg-screw |
|---------|--------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| rgbd+gp | dauntless-breeze-2 | 66.67% (8/12) | 75.00% (6/8) | 33.33% (2/6) | 100.00% (2/2) | 100.00% (2/2) | 100.00% (2/2) | 50.00% (1/2) | 100.00% (1/1) | 100.00% (1/1) |
| rgbd+skill | misunderstood-firebrand-6 | 91.67% (11/12) | 45.45% (5/11) | 40.00% (2/5) | 100.00% (2/2) | 0.00% (0/1) | 100.00% (1/1) | 100.00% (1/1) | 0.00% (0/1) | — |
| rgbd+gp+skill | vocal-bush-11 | 83.33% (10/12) | 40.00% (4/10) | 25.00% (1/4) | 100.00% (1/1) | 100.00% (1/1) | 0.00% (0/1) | — | — | — |
| rgbd+colored gp | gentle-fog-7 | 91.67% (11/12) | 63.64% (7/11) | 42.86% (3/7) | 100.00% (3/3) | 100.00% (3/3) | 100.00% (3/3) | 33.33% (1/3) | 100.00% (1/1) | 100.00% (1/1) |
| rgbd | breezy-rain-3 | 83.33% (10/12) | 50.00% (5/10) | 60.00% (3/5) | 100.00% (3/3) | 33.33% (1/3) | 100.00% (1/1) | 0.00% (0/1) | — | — |
| rgb | fiery-snowball-4 | 100.00% (12/12) | 58.33% (7/12) | 28.57% (2/7) | 100.00% (2/2) | 100.00% (2/2) | 50.00% (1/2) | 100.00% (1/1) | 0.00% (0/1) | — |

> `—` = no rollout reached this skill. Data source: `skill_state_counts` / `skill_completion_counts` in `logs/evaluate_model/round_table/<checkpoint_name>/*.json`, cross-validated against stdout `Skill success rates`.


---
## Appendix A: Checkpoint Config Verification

### Multi-task BC (one_leg+round_table+lamp, 3×100 traj, low, DiT)

| Condition | RUN_ID | Project | Local file | suffix | GP→obs | skill→obs | colored→obs | epochs |
|---------|--------|--------|---------|--------|:---:|:---:|:---:|:---:|
| rgbd+gp | autumn-dust-13 | 0610 | `...0610_autumn-dust-13_latest_3000.pt` | rgbd-skill | True | False | False | 3000 |
| rgbd+skill | iconic-surf-2 | 0526 | `...0526_iconic-surf-2_last_.pt` | rgbd-only-skill | False | True | False | 3000 |
| ~~rgbd+skill~~ | ~~icy-vortex-9~~ | 0610 | `...0610_icy-vortex-9_latest_3000.pt` | rgbd-skill | **True** ❌ | **False** ❌ | False | 3000 |
| rgbd+gp+skill | fresh-tree-11 | 0610 | `...0610_fresh-tree-11_latest_3000.pt` | rgbd-skill | True | True | False | 3000 |
| rgbd+colored gp | absurd-voice-2 | 0610 | `...0610_absurd-voice-2_latest_3000.pt` | rgbd-skill-colored | True | False | True | 3000 |
| rgbd | clear-water-12 | 0610 | `...0610_clear-water-12_latest_3000.pt` | rgbd | False | False | False | 3000 |
| rgb | true-firefly-8 | 0610 | `...0610_true-firefly-8_latest_3000.pt` | rgbd | False | False | False | 3000 |

> **icy-vortex-9**: Config shows GP=True, skill=False — identical to autumn-dust-13 (rgbd+gp). Notion labeled "rgbd+skill" but actual training used GP without skill. **Skipped from eval.**

### Single-task BC (round_table, 200 traj, low, DiT)

| Condition | RUN_ID | Project | Local file | suffix | GP→obs | skill→obs | colored→obs |
|---------|--------|--------|---------|--------|:---:|:---:|:---:|
| rgbd+gp | dauntless-breeze-2 | 0428 | `...0428_dauntless-breeze-2_last_.pt` | rgbd-skill | True | False | False * |
| rgbd+skill | misunderstood-firebrand-6 | 0428 | `...0428_misunderstood-firebrand-6_last_.pt` | rgbd-only-skill | False | True | False * |
| rgbd+gp+skill | vocal-bush-11 | 500 | `...500_vocal-bush-11_last_.pt` | rgbd-skill | True † | True † | False * |
| rgbd+colored gp | gentle-fog-7 | 0428 | `...0428_gentle-fog-7_last_.pt` | rgbd-skill-colored | True | False | True † |
| rgbd | breezy-rain-3 | 0428 | `...0428_breezy-rain-3_last_.pt` | rgbd | False | False | False * |
| rgb | fiery-snowball-4 | 0428 | `...0428_fiery-snowball-4_last_.pt` | rgbd | False | False | False * |

> \* `annotate_guidance_point_colored` key missing (old format) — defaults to False (correct, non-colored).
> † Key missing in original checkpoint — **manually added** based on data suffix evidence. See §Config Fixes.

### RPPO (single-task, low, state-based)

| Task | Local file | obs_type | randomness | iteration | train SR |
|------|---------|------|-----------|:---:|:---:|
| one_leg | `checkpoints/rppo/one_leg/low/actor_chkpt.pt` | state | low | 306 | 97.5% |
| round_table | `checkpoints/rppo/round_table/low/actor_chkpt.pt` | state | low | 921 | 96.0% |
| lamp | `checkpoints/rppo/lamp/low/actor_chkpt.pt` | state | low | 571 | 98.2% |

### Config Fixes Applied

Two single-task BC checkpoints had missing config keys (old format, predating the `annotate_*` config flags):

| RUN_ID | Missing keys | Added values | Rationale |
|--------|-------------|--------------|-----------|
| gentle-fog-7 | `annotate_guidance_point_colored` | True | suffix=rgbd-skill-colored, GP colored in data |
| vocal-bush-11 | `annotate_guidance_point` | True | suffix=rgbd-skill, GP baked in data |
| | `annotate_skill_one_hot` | True | skill_dim=5 confirms skill was active |
| | `annotate_guidance_point_colored` | False | not colored variant |

Backups: `*_last_.pt.bak` in same directory.
