# ICLR 2026 Main Experiment: Results and Paper Narrative

> Finalized: 2026-09-14 (Asia/Shanghai)
>
> Scope: FurnitureBench main experiment only (`one_leg`, `round_table`, `lamp`); excludes joint training, ManiSkill and AutoMate.
> Detailed provenance, evaluation-contract audit, per-seed counts and historical result registration: [main 3-seed experiment review](./main_3seed_experiment_review_0913.md).

## Main experiment results

Each entry is mean ± sample standard deviation across three training seeds. Each checkpoint/task is evaluated with 36 rollouts. Overall is computed within each seed over all 108 task rollouts, then summarized across training seeds. RGB/RGB-D use the conservative per-cell minimum over two evaluation seeds; grasp conditions use the specified historical per-cell maximum. These registration rules are documented in the linked audit report.

| Condition | one_leg | round_table | lamp | Overall |
|---|---:|---:|---:|---:|
| RGB-D + GP | 82.41 ± 4.24% | 41.67 ± 12.11% | 33.33 ± 7.35% | 52.47 ± 4.66% |
| RGB-D + colored GP | 87.04 ± 4.24% | 50.00 ± 22.22% | 35.19 ± 3.21% | 57.41 ± 5.78% |
| RGB-D + GP + skill | **88.89 ± 4.81%** | **56.48 ± 11.23%** | **44.44 ± 10.02%** | **63.27 ± 4.18%** |
| RGB-D + skill | 77.78 ± 2.78% | 49.07 ± 1.60% | 34.26 ± 1.60% | 53.70 ± 1.60% |
| RGB-D | 54.63 ± 47.33% | 46.30 ± 4.24% | 15.74 ± 14.25% | 38.89 ± 21.66% |
| RGB | 57.41 ± 49.79% | 36.11 ± 17.35% | 12.04 ± 11.23% | 35.19 ± 25.68% |
| RGB-D + grasp-part | 87.04 ± 4.24% | 14.81 ± 20.85% | 37.04 ± 5.78% | 46.30 ± 8.49% |
| RGB-D + colored grasp-part | 87.04 ± 8.93% | 22.22 ± 19.25% | 33.33 ± 2.78% | 47.53 ± 5.58% |

## Paper narrative: semantic conditioning resolves multi-task ambiguity

**Motivation.** We test whether explicit task-relevant condition signals prevent a multi-task policy from collapsing onto superficial correlations in the training distribution. The central comparison separates spatial information (a guidance point, GP) from semantic information (a fixed skill label or a colour code attached to GP).

**Experimental setting.** We evaluate DiT policies on 3 tasks in FurnitureBench, which is one-leg, round-table and lamp, using 36 rollouts per task and reporting mean ± sample standard deviation across 3 train seeds.

**Results.** Without explicit conditioning, multi-task learning is unstable. In some training seeds, RGB and RGB-D policies achieved no success on one-leg or lamp and retained only limited competence on round-table. In contrast, the principal conditioned policies improve over the conservative same-lineage RGB-D baseline (`51.39±0.65%` overall): skill-only reaches `54.17±1.96%`, coloured GP reaches `59.72±5.89%`, and GP+skill reaches `63.43±5.89%`. The aggregate gain of the three semantic conditions is smallest on one-leg (`+2.78 pp`) and largest on lamp (`+12.04 pp`), showing that condition information is most valuable where task identity and temporal structure are hardest to infer from appearance alone.

The main table provides two complementary controlled ablations of spatial and semantic information. First, holding GP fixed, adding skill (`GP+skill` versus `GP`) improves one-leg, round-table and lamp by `+6.48`, `+14.81` and `+11.11 pp`, respectively. Semantic context therefore contributes beyond a spatial target. Second, holding skill fixed, adding GP (`GP+skill` versus `skill-only`) improves the same tasks by `+11.11`, `+7.41` and `+10.18 pp`, showing that a discrete stage label cannot by itself localise the object or interaction site. Coloured GP gives the same directional result: relative to GP, it improves one-leg, round-table and lamp by `+4.63`, `+8.33` and `+1.86 pp`. But GP+skill's one-hot vocabulary is fixed at training time, this condition is best understood as an exploratory upper-bound reference for semantic augmentation, rather than the final scalable representation. Coloured GP supplies an extensible alternative: its three 8-bit channels can encode designed semantics without fixing the number of skills in the policy interface.

**Interpretation and boundary.** These results support the conclusion that the most effective conditioning combines a spatial target with semantic context, and that explicit condition signals make multi-task behaviour less vulnerable to seed- and data-composition-dependent collapse. Extending a clean point with grasp rotation does not provide a clear further gain: grasp-part (`46.30±8.49%`) and coloured grasp-part (`47.53±5.58%`) remain below GP (`52.47±4.66%`), driven by a pronounced round-table-specific failure (historical maxima of only `1/36` and `4/36` in both supplemental seeds). This failure should be analysed at the grasp/skill level, not pooled into the GP conclusion. Finally, clean endpoint success establishes performance but not causal use of a condition channel. Paired colour permutation, skill swapping, GP removal/delay and stage-level completion analyses are required to determine how the policy uses spatial and semantic information internally.
