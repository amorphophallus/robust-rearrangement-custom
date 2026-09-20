# Figure 1 样板说明

这三套样板都保留了论文当前固定的三项职责：家具拼装难点、conditioning interface 设计空间，以及 VLM→interface→shared DiT action expert 的系统关系。它们不展示成功率数字，以免 Introduction figure 与 Results figures 重复。

## Sample A — Triptych

- 经典 `problem → interface taxonomy → system` 三联图。
- 信息最完整，读者第一次看最容易理解。
- 缺点是 panel b 的接口表占据较多空间，视觉上略密。

### Scene-based teaser draft v1

- `figure1-teaser-generated-v1.png` 沿用 Sample A 的三联职责，但以家具拼装场景序列、condition overlay、VLM/DiT 图标和闭环 refresh 路径替代大部分方框。
- panel a 解释 action error 经状态转移传播与 guidance 周期重锚的区别；panel b 对比五类 conditioning interfaces；panel c 展示 VLM--TAGPoint--DiT 的端到端关系。
- 该图是构图和视觉风格稿，不是可直接投稿的最终证据图。定稿时应把生成的机器人场景换成真实 FurnitureBench/真机截图，并在 draw.io 中重绘文字、箭头和 interface overlays 后导出矢量 PDF。

## Sample B — Anchor-centered

- 强调长时程动作误差会改变后续状态，而当前图像重新 grounding 的 target 是外部参照。
- TAGPoint 位于两个系统的中央，双系统分工最直观。
- panel c 同时把全文 evidence ladder 展示出来，适合分析型论文。

## Sample C — Question-first

- 最贴合标题 `What Should a VLM Tell an Action Expert?`。
- panel b 是视觉中心，突出本文比较五种 conditioning interfaces，而不是宣称新架构。
- panel c 强调相同 action expert 下的 controlled comparison，与 Much Ado About Noising 式的论证型论文最接近。

## 建议

优先在 A 与 C 中选择：

- 希望读者快速理解完整系统，选 A；
- 希望强化“比较接口而非提出新架构”的论文定位，选 C；
- 希望把“不递归累积的 target reference”作为最醒目的新故事，选 B。

选定后再把示意框替换为真实 FurnitureBench/真机图像，并统一为 TAGPoint 正式术语。
