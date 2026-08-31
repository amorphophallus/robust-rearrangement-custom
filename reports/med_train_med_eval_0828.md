# Med Train / Med Eval Report

## Protocol

- Training project: `multi-task-rgbd-skill-med-0801`.
- Evaluation randomness: existing `med` reset randomization; no fixed eval seed or state bank.
- Matrix: 8 checkpoints x 3 tasks x 36 rollouts = 864 rollouts.
- Runtime: local RTX 3060, 3 parallel environments, 1000 maximum steps per rollout.
- Artifacts: first 3 videos per checkpoint-task; no rollout pickle and no depth export.
- Step conditional rate is completed/reached. Step cumulative rate is completed/36.

## Overall Results

| Exp | Condition | Checkpoint | SHA256 | one_leg | round_table | lamp | All tasks |
|---:|---|---|---|---:|---:|---:|---:|
| 1 | rgbd | `jolly-frost-15` | `b2ae2e2c044d` | 55.6% (20/36) | 2.8% (1/36) | 8.3% (3/36) | 22.2% (24/108) |
| 2 | rgbd_guidance_point | `smooth-snowflake-9` | `7fc7db0c2f36` | 61.1% (22/36) | 8.3% (3/36) | 25.0% (9/36) | 31.5% (34/108) |
| 3 | rgbd_colored_guidance_point | `honest-cherry-19` | `f0de9e5fcbf8` | 75.0% (27/36) | 5.6% (2/36) | 19.4% (7/36) | 33.3% (36/108) |
| 4 | rgbd_skill_one_hot | `wandering-wind-16` | `4a57c5010968` | 50.0% (18/36) | 13.9% (5/36) | 27.8% (10/36) | 30.6% (33/108) |
| 5 | rgbd_guidance_point_skill_one_hot | `polar-silence-10` | `190224ee8b55` | 58.3% (21/36) | 33.3% (12/36) | 13.9% (5/36) | 35.2% (38/108) |
| 6 | rgb_baseline | `efficient-violet-5` | `ef2672b66382` | 72.2% (26/36) | 0.0% (0/36) | 22.2% (8/36) | 31.5% (34/108) |
| 7 | rgbd_grasp_part | `devoted-glade-18` | `3a3899de4e7e` | 47.2% (17/36) | 5.6% (2/36) | 33.3% (12/36) | 28.7% (31/108) |
| 8 | rgbd_colored_grasp_part | `major-salad-20` | `71617fe52ea1` | 63.9% (23/36) | 8.3% (3/36) | 19.4% (7/36) | 30.6% (33/108) |

## Assembly-Step Results

### one_leg

| Exp | Condition | top-leg |
|---:|---|---:|
| 1 | rgbd | 55.6% (20/36); cond. 55.6% (20/36) |
| 2 | rgbd_guidance_point | 61.1% (22/36); cond. 61.1% (22/36) |
| 3 | rgbd_colored_guidance_point | 75.0% (27/36); cond. 75.0% (27/36) |
| 4 | rgbd_skill_one_hot | 50.0% (18/36); cond. 50.0% (18/36) |
| 5 | rgbd_guidance_point_skill_one_hot | 58.3% (21/36); cond. 58.3% (21/36) |
| 6 | rgb_baseline | 72.2% (26/36); cond. 72.2% (26/36) |
| 7 | rgbd_grasp_part | 47.2% (17/36); cond. 47.2% (17/36) |
| 8 | rgbd_colored_grasp_part | 63.9% (23/36); cond. 63.9% (23/36) |

### round_table

| Exp | Condition | top-leg | leg-base |
|---:|---|---:|---:|
| 1 | rgbd | 38.9% (14/36); cond. 38.9% (14/36) | 2.8% (1/36); cond. 7.1% (1/14) |
| 2 | rgbd_guidance_point | 41.7% (15/36); cond. 41.7% (15/36) | 8.3% (3/36); cond. 20.0% (3/15) |
| 3 | rgbd_colored_guidance_point | 52.8% (19/36); cond. 52.8% (19/36) | 5.6% (2/36); cond. 10.5% (2/19) |
| 4 | rgbd_skill_one_hot | 33.3% (12/36); cond. 33.3% (12/36) | 13.9% (5/36); cond. 41.7% (5/12) |
| 5 | rgbd_guidance_point_skill_one_hot | 58.3% (21/36); cond. 58.3% (21/36) | 33.3% (12/36); cond. 57.1% (12/21) |
| 6 | rgb_baseline | 19.4% (7/36); cond. 19.4% (7/36) | 0.0% (0/36); cond. 0.0% (0/7) |
| 7 | rgbd_grasp_part | 41.7% (15/36); cond. 41.7% (15/36) | 5.6% (2/36); cond. 13.3% (2/15) |
| 8 | rgbd_colored_grasp_part | 47.2% (17/36); cond. 47.2% (17/36) | 8.3% (3/36); cond. 17.6% (3/17) |

### lamp

| Exp | Condition | base-bulb | base-hood |
|---:|---|---:|---:|
| 1 | rgbd | 11.1% (4/36); cond. 11.1% (4/36) | 8.3% (3/36); cond. 75.0% (3/4) |
| 2 | rgbd_guidance_point | 27.8% (10/36); cond. 27.8% (10/36) | 19.4% (7/36); cond. 87.5% (7/8) |
| 3 | rgbd_colored_guidance_point | 30.6% (11/36); cond. 30.6% (11/36) | 13.9% (5/36); cond. 55.6% (5/9) |
| 4 | rgbd_skill_one_hot | 30.6% (11/36); cond. 30.6% (11/36) | 19.4% (7/36); cond. 87.5% (7/8) |
| 5 | rgbd_guidance_point_skill_one_hot | 13.9% (5/36); cond. 13.9% (5/36) | 11.1% (4/36); cond. 100.0% (4/4) |
| 6 | rgb_baseline | 22.2% (8/36); cond. 22.2% (8/36) | 16.7% (6/36); cond. 100.0% (6/6) |
| 7 | rgbd_grasp_part | 33.3% (12/36); cond. 33.3% (12/36) | 30.6% (11/36); cond. 100.0% (11/11) |
| 8 | rgbd_colored_grasp_part | 30.6% (11/36); cond. 30.6% (11/36) | 8.3% (3/36); cond. 42.9% (3/7) |

## Skill-State Results

### one_leg

**Exp 1: rgbd**

| Skill state | Reached | Completed | Conditional SR | Cumulative completion |
|---|---:|---:|---:|---:|
| top-leg-pick | 36 | 36 | 100.0% (36/36) | 100.0% (36/36) |
| top-leg-push | 36 | 35 | 97.2% (35/36) | 97.2% (35/36) |
| leg-top-pick | 35 | 29 | 82.9% (29/35) | 80.6% (29/36) |
| leg-top-place | 29 | 21 | 72.4% (21/29) | 58.3% (21/36) |
| leg-top-insert | 21 | 20 | 95.2% (20/21) | 55.6% (20/36) |
| leg-top-screw | 18 | 18 | 100.0% (18/18) | 50.0% (18/36) |

**Exp 2: rgbd_guidance_point**

| Skill state | Reached | Completed | Conditional SR | Cumulative completion |
|---|---:|---:|---:|---:|
| top-leg-pick | 36 | 36 | 100.0% (36/36) | 100.0% (36/36) |
| top-leg-push | 36 | 33 | 91.7% (33/36) | 91.7% (33/36) |
| leg-top-pick | 33 | 30 | 90.9% (30/33) | 83.3% (30/36) |
| leg-top-place | 30 | 23 | 76.7% (23/30) | 63.9% (23/36) |
| leg-top-insert | 23 | 22 | 95.7% (22/23) | 61.1% (22/36) |
| leg-top-screw | 19 | 19 | 100.0% (19/19) | 52.8% (19/36) |

**Exp 3: rgbd_colored_guidance_point**

| Skill state | Reached | Completed | Conditional SR | Cumulative completion |
|---|---:|---:|---:|---:|
| top-leg-pick | 36 | 36 | 100.0% (36/36) | 100.0% (36/36) |
| top-leg-push | 36 | 35 | 97.2% (35/36) | 97.2% (35/36) |
| leg-top-pick | 35 | 31 | 88.6% (31/35) | 86.1% (31/36) |
| leg-top-place | 31 | 29 | 93.5% (29/31) | 80.6% (29/36) |
| leg-top-insert | 29 | 27 | 93.1% (27/29) | 75.0% (27/36) |
| leg-top-screw | 25 | 25 | 100.0% (25/25) | 69.4% (25/36) |

**Exp 4: rgbd_skill_one_hot**

| Skill state | Reached | Completed | Conditional SR | Cumulative completion |
|---|---:|---:|---:|---:|
| top-leg-pick | 36 | 36 | 100.0% (36/36) | 100.0% (36/36) |
| top-leg-push | 36 | 31 | 86.1% (31/36) | 86.1% (31/36) |
| leg-top-pick | 30 | 27 | 90.0% (27/30) | 75.0% (27/36) |
| leg-top-place | 27 | 19 | 70.4% (19/27) | 52.8% (19/36) |
| leg-top-insert | 18 | 16 | 88.9% (16/18) | 44.4% (16/36) |
| leg-top-screw | 15 | 15 | 100.0% (15/15) | 41.7% (15/36) |

**Exp 5: rgbd_guidance_point_skill_one_hot**

| Skill state | Reached | Completed | Conditional SR | Cumulative completion |
|---|---:|---:|---:|---:|
| top-leg-pick | 36 | 35 | 97.2% (35/36) | 97.2% (35/36) |
| top-leg-push | 35 | 34 | 97.1% (34/35) | 94.4% (34/36) |
| leg-top-pick | 34 | 29 | 85.3% (29/34) | 80.6% (29/36) |
| leg-top-place | 29 | 22 | 75.9% (22/29) | 61.1% (22/36) |
| leg-top-insert | 22 | 21 | 95.5% (21/22) | 58.3% (21/36) |
| leg-top-screw | 18 | 18 | 100.0% (18/18) | 50.0% (18/36) |

**Exp 6: rgb_baseline**

| Skill state | Reached | Completed | Conditional SR | Cumulative completion |
|---|---:|---:|---:|---:|
| top-leg-pick | 36 | 36 | 100.0% (36/36) | 100.0% (36/36) |
| top-leg-push | 36 | 35 | 97.2% (35/36) | 97.2% (35/36) |
| leg-top-pick | 35 | 34 | 97.1% (34/35) | 94.4% (34/36) |
| leg-top-place | 34 | 28 | 82.4% (28/34) | 77.8% (28/36) |
| leg-top-insert | 26 | 24 | 92.3% (24/26) | 66.7% (24/36) |
| leg-top-screw | 21 | 21 | 100.0% (21/21) | 58.3% (21/36) |

**Exp 7: rgbd_grasp_part**

| Skill state | Reached | Completed | Conditional SR | Cumulative completion |
|---|---:|---:|---:|---:|
| top-leg-pick | 36 | 31 | 86.1% (31/36) | 86.1% (31/36) |
| top-leg-push | 31 | 25 | 80.6% (25/31) | 69.4% (25/36) |
| leg-top-pick | 25 | 23 | 92.0% (23/25) | 63.9% (23/36) |
| leg-top-place | 23 | 21 | 91.3% (21/23) | 58.3% (21/36) |
| leg-top-insert | 21 | 18 | 85.7% (18/21) | 50.0% (18/36) |
| leg-top-screw | 15 | 14 | 93.3% (14/15) | 38.9% (14/36) |

**Exp 8: rgbd_colored_grasp_part**

| Skill state | Reached | Completed | Conditional SR | Cumulative completion |
|---|---:|---:|---:|---:|
| top-leg-pick | 36 | 35 | 97.2% (35/36) | 97.2% (35/36) |
| top-leg-push | 35 | 33 | 94.3% (33/35) | 91.7% (33/36) |
| leg-top-pick | 33 | 33 | 100.0% (33/33) | 91.7% (33/36) |
| leg-top-place | 33 | 26 | 78.8% (26/33) | 72.2% (26/36) |
| leg-top-insert | 26 | 23 | 88.5% (23/26) | 63.9% (23/36) |
| leg-top-screw | 23 | 23 | 100.0% (23/23) | 63.9% (23/36) |

### round_table

**Exp 1: rgbd**

| Skill state | Reached | Completed | Conditional SR | Cumulative completion |
|---|---:|---:|---:|---:|
| top-leg-push | 36 | 36 | 100.0% (36/36) | 100.0% (36/36) |
| leg-top-pick | 36 | 28 | 77.8% (28/36) | 77.8% (28/36) |
| leg-top-place | 28 | 17 | 60.7% (17/28) | 47.2% (17/36) |
| leg-top-insert | 17 | 16 | 94.1% (16/17) | 44.4% (16/36) |
| leg-top-screw | 16 | 14 | 87.5% (14/16) | 38.9% (14/36) |
| base-leg-pick | 14 | 5 | 35.7% (5/14) | 13.9% (5/36) |
| base-leg-place | 4 | 0 | 0.0% (0/4) | 0.0% (0/36) |
| base-leg-insert | 1 | 1 | 100.0% (1/1) | 2.8% (1/36) |
| base-leg-screw | 1 | 1 | 100.0% (1/1) | 2.8% (1/36) |

**Exp 2: rgbd_guidance_point**

| Skill state | Reached | Completed | Conditional SR | Cumulative completion |
|---|---:|---:|---:|---:|
| top-leg-push | 36 | 35 | 97.2% (35/36) | 97.2% (35/36) |
| leg-top-pick | 35 | 25 | 71.4% (25/35) | 69.4% (25/36) |
| leg-top-place | 25 | 16 | 64.0% (16/25) | 44.4% (16/36) |
| leg-top-insert | 16 | 16 | 100.0% (16/16) | 44.4% (16/36) |
| leg-top-screw | 16 | 15 | 93.8% (15/16) | 41.7% (15/36) |
| base-leg-pick | 15 | 11 | 73.3% (11/15) | 30.6% (11/36) |
| base-leg-place | 11 | 5 | 45.5% (5/11) | 13.9% (5/36) |
| base-leg-insert | 5 | 4 | 80.0% (4/5) | 11.1% (4/36) |
| base-leg-screw | 4 | 3 | 75.0% (3/4) | 8.3% (3/36) |

**Exp 3: rgbd_colored_guidance_point**

| Skill state | Reached | Completed | Conditional SR | Cumulative completion |
|---|---:|---:|---:|---:|
| top-leg-push | 36 | 34 | 94.4% (34/36) | 94.4% (34/36) |
| leg-top-pick | 34 | 29 | 85.3% (29/34) | 80.6% (29/36) |
| leg-top-place | 29 | 23 | 79.3% (23/29) | 63.9% (23/36) |
| leg-top-insert | 23 | 23 | 100.0% (23/23) | 63.9% (23/36) |
| leg-top-screw | 22 | 18 | 81.8% (18/22) | 50.0% (18/36) |
| base-leg-pick | 19 | 6 | 31.6% (6/19) | 16.7% (6/36) |
| base-leg-place | 6 | 3 | 50.0% (3/6) | 8.3% (3/36) |
| base-leg-insert | 3 | 2 | 66.7% (2/3) | 5.6% (2/36) |
| base-leg-screw | 2 | 2 | 100.0% (2/2) | 5.6% (2/36) |

**Exp 4: rgbd_skill_one_hot**

| Skill state | Reached | Completed | Conditional SR | Cumulative completion |
|---|---:|---:|---:|---:|
| top-leg-push | 36 | 35 | 97.2% (35/36) | 97.2% (35/36) |
| leg-top-pick | 35 | 28 | 80.0% (28/35) | 77.8% (28/36) |
| leg-top-place | 28 | 13 | 46.4% (13/28) | 36.1% (13/36) |
| leg-top-insert | 13 | 12 | 92.3% (12/13) | 33.3% (12/36) |
| leg-top-screw | 12 | 12 | 100.0% (12/12) | 33.3% (12/36) |
| base-leg-pick | 12 | 9 | 75.0% (9/12) | 25.0% (9/36) |
| base-leg-place | 9 | 7 | 77.8% (7/9) | 19.4% (7/36) |
| base-leg-insert | 7 | 7 | 100.0% (7/7) | 19.4% (7/36) |
| base-leg-screw | 7 | 5 | 71.4% (5/7) | 13.9% (5/36) |

**Exp 5: rgbd_guidance_point_skill_one_hot**

| Skill state | Reached | Completed | Conditional SR | Cumulative completion |
|---|---:|---:|---:|---:|
| top-leg-push | 36 | 36 | 100.0% (36/36) | 100.0% (36/36) |
| leg-top-pick | 36 | 33 | 91.7% (33/36) | 91.7% (33/36) |
| leg-top-place | 33 | 21 | 63.6% (21/33) | 58.3% (21/36) |
| leg-top-insert | 21 | 21 | 100.0% (21/21) | 58.3% (21/36) |
| leg-top-screw | 21 | 21 | 100.0% (21/21) | 58.3% (21/36) |
| base-leg-pick | 21 | 15 | 71.4% (15/21) | 41.7% (15/36) |
| base-leg-place | 15 | 14 | 93.3% (14/15) | 38.9% (14/36) |
| base-leg-insert | 14 | 14 | 100.0% (14/14) | 38.9% (14/36) |
| base-leg-screw | 14 | 12 | 85.7% (12/14) | 33.3% (12/36) |

**Exp 6: rgb_baseline**

| Skill state | Reached | Completed | Conditional SR | Cumulative completion |
|---|---:|---:|---:|---:|
| top-leg-push | 36 | 34 | 94.4% (34/36) | 94.4% (34/36) |
| leg-top-pick | 34 | 21 | 61.8% (21/34) | 58.3% (21/36) |
| leg-top-place | 21 | 9 | 42.9% (9/21) | 25.0% (9/36) |
| leg-top-insert | 9 | 9 | 100.0% (9/9) | 25.0% (9/36) |
| leg-top-screw | 9 | 7 | 77.8% (7/9) | 19.4% (7/36) |
| base-leg-pick | 7 | 2 | 28.6% (2/7) | 5.6% (2/36) |
| base-leg-place | 2 | 0 | 0.0% (0/2) | 0.0% (0/36) |
| base-leg-insert | 0 | 0 | N/A | 0.0% (0/36) |
| base-leg-screw | 0 | 0 | N/A | 0.0% (0/36) |

**Exp 7: rgbd_grasp_part**

| Skill state | Reached | Completed | Conditional SR | Cumulative completion |
|---|---:|---:|---:|---:|
| top-leg-push | 36 | 36 | 100.0% (36/36) | 100.0% (36/36) |
| leg-top-pick | 36 | 30 | 83.3% (30/36) | 83.3% (30/36) |
| leg-top-place | 30 | 18 | 60.0% (18/30) | 50.0% (18/36) |
| leg-top-insert | 18 | 18 | 100.0% (18/18) | 50.0% (18/36) |
| leg-top-screw | 18 | 15 | 83.3% (15/18) | 41.7% (15/36) |
| base-leg-pick | 15 | 7 | 46.7% (7/15) | 19.4% (7/36) |
| base-leg-place | 7 | 3 | 42.9% (3/7) | 8.3% (3/36) |
| base-leg-insert | 3 | 2 | 66.7% (2/3) | 5.6% (2/36) |
| base-leg-screw | 2 | 2 | 100.0% (2/2) | 5.6% (2/36) |

**Exp 8: rgbd_colored_grasp_part**

| Skill state | Reached | Completed | Conditional SR | Cumulative completion |
|---|---:|---:|---:|---:|
| top-leg-push | 36 | 36 | 100.0% (36/36) | 100.0% (36/36) |
| leg-top-pick | 36 | 30 | 83.3% (30/36) | 83.3% (30/36) |
| leg-top-place | 30 | 20 | 66.7% (20/30) | 55.6% (20/36) |
| leg-top-insert | 20 | 20 | 100.0% (20/20) | 55.6% (20/36) |
| leg-top-screw | 20 | 17 | 85.0% (17/20) | 47.2% (17/36) |
| base-leg-pick | 17 | 7 | 41.2% (7/17) | 19.4% (7/36) |
| base-leg-place | 6 | 2 | 33.3% (2/6) | 5.6% (2/36) |
| base-leg-insert | 3 | 3 | 100.0% (3/3) | 8.3% (3/36) |
| base-leg-screw | 3 | 3 | 100.0% (3/3) | 8.3% (3/36) |

### lamp

**Exp 1: rgbd**

| Skill state | Reached | Completed | Conditional SR | Cumulative completion |
|---|---:|---:|---:|---:|
| base-bulb-push | 36 | 33 | 91.7% (33/36) | 91.7% (33/36) |
| bulb-base-pick | 33 | 32 | 97.0% (32/33) | 88.9% (32/36) |
| bulb-base-place | 32 | 12 | 37.5% (12/32) | 33.3% (12/36) |
| bulb-base-insert | 12 | 12 | 100.0% (12/12) | 33.3% (12/36) |
| bulb-base-screw | 12 | 4 | 33.3% (4/12) | 11.1% (4/36) |
| hood-base-pick | 4 | 3 | 75.0% (3/4) | 8.3% (3/36) |
| hood-base-place | 3 | 3 | 100.0% (3/3) | 8.3% (3/36) |

**Exp 2: rgbd_guidance_point**

| Skill state | Reached | Completed | Conditional SR | Cumulative completion |
|---|---:|---:|---:|---:|
| base-bulb-push | 36 | 32 | 88.9% (32/36) | 88.9% (32/36) |
| bulb-base-pick | 32 | 30 | 93.8% (30/32) | 83.3% (30/36) |
| bulb-base-place | 30 | 20 | 66.7% (20/30) | 55.6% (20/36) |
| bulb-base-insert | 20 | 20 | 100.0% (20/20) | 55.6% (20/36) |
| bulb-base-screw | 20 | 10 | 50.0% (10/20) | 27.8% (10/36) |
| hood-base-pick | 8 | 8 | 100.0% (8/8) | 22.2% (8/36) |
| hood-base-place | 8 | 7 | 87.5% (7/8) | 19.4% (7/36) |

**Exp 3: rgbd_colored_guidance_point**

| Skill state | Reached | Completed | Conditional SR | Cumulative completion |
|---|---:|---:|---:|---:|
| base-bulb-push | 36 | 30 | 83.3% (30/36) | 83.3% (30/36) |
| bulb-base-pick | 30 | 29 | 96.7% (29/30) | 80.6% (29/36) |
| bulb-base-place | 29 | 15 | 51.7% (15/29) | 41.7% (15/36) |
| bulb-base-insert | 15 | 15 | 100.0% (15/15) | 41.7% (15/36) |
| bulb-base-screw | 15 | 11 | 73.3% (11/15) | 30.6% (11/36) |
| hood-base-pick | 9 | 6 | 66.7% (6/9) | 16.7% (6/36) |
| hood-base-place | 6 | 5 | 83.3% (5/6) | 13.9% (5/36) |

**Exp 4: rgbd_skill_one_hot**

| Skill state | Reached | Completed | Conditional SR | Cumulative completion |
|---|---:|---:|---:|---:|
| base-bulb-push | 36 | 33 | 91.7% (33/36) | 91.7% (33/36) |
| bulb-base-pick | 33 | 33 | 100.0% (33/33) | 91.7% (33/36) |
| bulb-base-place | 33 | 15 | 45.5% (15/33) | 41.7% (15/36) |
| bulb-base-insert | 15 | 15 | 100.0% (15/15) | 41.7% (15/36) |
| bulb-base-screw | 15 | 11 | 73.3% (11/15) | 30.6% (11/36) |
| hood-base-pick | 8 | 7 | 87.5% (7/8) | 19.4% (7/36) |
| hood-base-place | 7 | 7 | 100.0% (7/7) | 19.4% (7/36) |

**Exp 5: rgbd_guidance_point_skill_one_hot**

| Skill state | Reached | Completed | Conditional SR | Cumulative completion |
|---|---:|---:|---:|---:|
| base-bulb-push | 36 | 31 | 86.1% (31/36) | 86.1% (31/36) |
| bulb-base-pick | 31 | 27 | 87.1% (27/31) | 75.0% (27/36) |
| bulb-base-place | 27 | 10 | 37.0% (10/27) | 27.8% (10/36) |
| bulb-base-insert | 10 | 10 | 100.0% (10/10) | 27.8% (10/36) |
| bulb-base-screw | 9 | 4 | 44.4% (4/9) | 11.1% (4/36) |
| hood-base-pick | 4 | 4 | 100.0% (4/4) | 11.1% (4/36) |
| hood-base-place | 4 | 4 | 100.0% (4/4) | 11.1% (4/36) |

**Exp 6: rgb_baseline**

| Skill state | Reached | Completed | Conditional SR | Cumulative completion |
|---|---:|---:|---:|---:|
| base-bulb-push | 36 | 32 | 88.9% (32/36) | 88.9% (32/36) |
| bulb-base-pick | 32 | 28 | 87.5% (28/32) | 77.8% (28/36) |
| bulb-base-place | 28 | 16 | 57.1% (16/28) | 44.4% (16/36) |
| bulb-base-insert | 16 | 16 | 100.0% (16/16) | 44.4% (16/36) |
| bulb-base-screw | 16 | 8 | 50.0% (8/16) | 22.2% (8/36) |
| hood-base-pick | 6 | 6 | 100.0% (6/6) | 16.7% (6/36) |
| hood-base-place | 6 | 6 | 100.0% (6/6) | 16.7% (6/36) |

**Exp 7: rgbd_grasp_part**

| Skill state | Reached | Completed | Conditional SR | Cumulative completion |
|---|---:|---:|---:|---:|
| base-bulb-push | 36 | 34 | 94.4% (34/36) | 94.4% (34/36) |
| bulb-base-pick | 34 | 31 | 91.2% (31/34) | 86.1% (31/36) |
| bulb-base-place | 31 | 18 | 58.1% (18/31) | 50.0% (18/36) |
| bulb-base-insert | 18 | 18 | 100.0% (18/18) | 50.0% (18/36) |
| bulb-base-screw | 18 | 12 | 66.7% (12/18) | 33.3% (12/36) |
| hood-base-pick | 11 | 11 | 100.0% (11/11) | 30.6% (11/36) |
| hood-base-place | 11 | 11 | 100.0% (11/11) | 30.6% (11/36) |

**Exp 8: rgbd_colored_grasp_part**

| Skill state | Reached | Completed | Conditional SR | Cumulative completion |
|---|---:|---:|---:|---:|
| base-bulb-push | 36 | 33 | 91.7% (33/36) | 91.7% (33/36) |
| bulb-base-pick | 33 | 32 | 97.0% (32/33) | 88.9% (32/36) |
| bulb-base-place | 32 | 18 | 56.2% (18/32) | 50.0% (18/36) |
| bulb-base-insert | 18 | 18 | 100.0% (18/18) | 50.0% (18/36) |
| bulb-base-screw | 18 | 11 | 61.1% (11/18) | 30.6% (11/36) |
| hood-base-pick | 7 | 4 | 57.1% (4/7) | 11.1% (4/36) |
| hood-base-place | 4 | 3 | 75.0% (3/4) | 8.3% (3/36) |

## Reproducibility and Assets

- Evaluation host: `lht-3060-12G`; GPU index: `0`.
- Repository commit at launch: `f62eff3a461e4b6f4ae4c1d2fa2e17c67abc7e6a`.
- Execution manifest: `/data/hy/robust-rearrangement/logs/med_train_med_eval_0828/formal_manifest.json`.
- Checkpoint and video SHA/path inventory is retained in the execution manifest and task summaries.

## Interpretation

These numbers compare one trained checkpoint per condition. They are not mean/max statistics over independent training seeds. Each checkpoint receives 36 independently randomized med initializations per task under the same distribution and rollout budget.

### Round-table failure concentration

Across all eight checkpoints, round-table succeeds on 28/288 rollouts (9.7%). The first `top-leg` assembly is completed on 120/288 rollouts (41.7%); only 28/120 rollouts (23.3%) that reach `leg-base` complete the second assembly. The pooled transition counts identify three main bottlenecks: `leg-top-place` completes on 137/224 reaches (61.2%), `base-leg-pick` on 62/120 (51.7%), and `base-leg-place` on 34/60 (56.7%). Insert and screw transitions are generally much stronger once reached, although their late-stage denominators are small.

| Round-table subtask | Reached | Completed | Conditional SR | Cumulative completion |
|---|---:|---:|---:|---:|
| `top-leg-push` | 288 | 282 | 97.9% (282/288) | 97.9% (282/288) |
| `leg-top-pick` | 282 | 224 | 79.4% (224/282) | 77.8% (224/288) |
| `leg-top-place` | 224 | 137 | 61.2% (137/224) | 47.6% (137/288) |
| `leg-top-insert` | 137 | 135 | 98.5% (135/137) | 46.9% (135/288) |
| `leg-top-screw` | 134 | 119 | 88.8% (119/134) | 41.3% (119/288) |
| `base-leg-pick` | 120 | 62 | 51.7% (62/120) | 21.5% (62/288) |
| `base-leg-place` | 60 | 34 | 56.7% (34/60) | 11.8% (34/288) |
| `base-leg-insert` | 36 | 33 | 91.7% (33/36) | 11.5% (33/288) |
| `base-leg-screw` | 33 | 28 | 84.8% (28/33) | 9.7% (28/288) |

Conditional SR is `completed/reached`; cumulative completion is `completed/288`. Adjacent reached/completed counts can differ slightly because the evaluator records each state independently at rollout termination and transition boundaries.

Manual review of representative retained videos shows full-length failures repeatedly contacting or approaching a part without establishing a stable grasp or placement, then timing out at 1000 steps. Exp5 demonstrates that the task is solvable under this protocol (12/36), and its combination of guidance point plus skill one-hot performs best on both assembly stages. This supports a long-horizon error-compounding and pick/place localization hypothesis more strongly than a universal insertion-controller failure. The video sample is qualitative only: at most the first three saved rollouts per cell were retained, so it is not an unbiased sample of all failures.

## Appendix A: Checkpoint Locations

| Exp | Condition | Run | Absolute checkpoint path |
|---:|---|---|---|
| 1 | rgbd | `jolly-frost-15` | `/data/hy/robust-rearrangement/checkpoints/bc/one_leg+round_table+lamp/low/multi-task-rgbd-skill-med-0801_jolly-frost-15_last_.pt` |
| 2 | rgbd_guidance_point | `smooth-snowflake-9` | `/data/hy/robust-rearrangement/checkpoints/bc/one_leg+round_table+lamp/low/multi-task-rgbd-skill-med-0801_smooth-snowflake-9_last_.pt` |
| 3 | rgbd_colored_guidance_point | `honest-cherry-19` | `/data/hy/robust-rearrangement/checkpoints/bc/one_leg+round_table+lamp/low/multi-task-rgbd-skill-med-0801_honest-cherry-19_last_.pt` |
| 4 | rgbd_skill_one_hot | `wandering-wind-16` | `/data/hy/robust-rearrangement/checkpoints/bc/one_leg+round_table+lamp/low/multi-task-rgbd-skill-med-0801_wandering-wind-16_last_.pt` |
| 5 | rgbd_guidance_point_skill_one_hot | `polar-silence-10` | `/data/hy/robust-rearrangement/checkpoints/bc/one_leg+round_table+lamp/low/multi-task-rgbd-skill-med-0801_polar-silence-10_last_.pt` |
| 6 | rgb_baseline | `efficient-violet-5` | `/data/hy/robust-rearrangement/checkpoints/bc/one_leg+round_table+lamp/low/multi-task-rgbd-skill-med-0801_efficient-violet-5_last_.pt` |
| 7 | rgbd_grasp_part | `devoted-glade-18` | `/data/hy/robust-rearrangement/checkpoints/bc/one_leg+round_table+lamp/low/multi-task-rgbd-skill-med-0801_devoted-glade-18_last_.pt` |
| 8 | rgbd_colored_grasp_part | `major-salad-20` | `/data/hy/robust-rearrangement/checkpoints/bc/one_leg+round_table+lamp/low/multi-task-rgbd-skill-med-0801_major-salad-20_last_.pt` |
