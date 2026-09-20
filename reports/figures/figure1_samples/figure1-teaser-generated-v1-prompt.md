# Figure 1 teaser v1 — generation prompt

Reference composition: `figure1-sample-a-triptych.png`. Use it only as the three-panel scaffold and replace the box-heavy visual language.

Create a polished, full-width scientific-paper teaser for an ICLR robotics paper. Use a wide, approximately 2:1 landscape canvas with three connected panels labelled a, b and c. Combine precise vector geometry with compact, semi-realistic 3D robot-assembly thumbnails on a white-to-pale-blue background. Use a restrained, colour-blind-safe navy, cyan/teal, coral and violet palette, subtle depth and crisp edges. Include meaningful camera views, robot/furniture scenes, target-point overlays, trajectories and state transitions; avoid a composition dominated by text boxes. Keep text minimal and large, with no numerical results, logos or watermark.

## Panel a — Why explicit guidance?

Show two sequences of a Franka-like robot assembling a furniture part. In the upper coral/red path, a small action error changes the part pose and the next observation; use three scene thumbnails, ghosted misalignment and a drifting trajectory to convey closed-loop propagation through state transitions. In the lower teal path, the current observation is refreshed, a VLM eye/camera icon re-detects the target and a coloured point is overlaid at the updated interaction location. A refresh arrow should communicate periodically re-anchored guidance without implying that physical failures are undone. Use only the short labels `Why explicit guidance?`, `Action error` and `Periodic re-grounding`.

## Panel b — Conditioning interfaces

Make this the visual centre. Present five compact interface cards using robot camera-view thumbnails and symbolic overlays: `Skill` as a categorical token without a location; `GP` as a plain point on the target part; `TAGPoint` as a subtly highlighted coloured point with a small gripper-mode glyph; `GP + skill` as the point plus categorical token; and `6D grasp` as a target marker plus a three-axis orientation triad. Add two small, desaturated alternatives at the upper edge: an ambiguous speech bubble for open-ended text and a point-cloud/metric-coordinate icon with calibration for 3D output. Use only the labels `Conditioning interfaces`, `Skill`, `GP`, `TAGPoint`, `GP + skill` and `6D grasp`.

## Panel c — VLM → TAGPoint → DiT

Use icons and scene thumbnails rather than boxes. A current RGB observation and instruction flow into a compact VLM eye/brain icon; the VLM outputs TAGPoint over an RGB-D furniture scene; this enters a compact transformer/action-expert icon; smooth multi-step action chunks drive the robot to a correctly assembled result. Add a thin loop from the updated observation to the VLM to indicate periodic refresh. Use only the labels `VLM → TAGPoint → DiT`, `VLM` and `DiT action expert`.

Allocate approximately 28%, 42% and 30% of the width to panels a, b and c. Connect the panels with consistent scene thumbnails and thin visual flow lines. Use spacing or faint dividers instead of heavy borders. The figure must remain interpretable at paper-column scale. Avoid excessive glow, decorative circuits, humanoid robots, childish icons, crowded annotations, illegible text and large empty areas.
