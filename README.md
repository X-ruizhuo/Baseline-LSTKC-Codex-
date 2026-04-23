# Baseline-LSTKC-Codex-

## 1.loss级融合

本次改动基于 LSTKC++ 的“长短期知识分解与巩固”思想，在不重构主干网络的前提下完成了低侵入融合，重点落在训练器与损失协同层，目标是在 16G 显存服务器上稳定训练。

### 改动概览

- 在 `Bi-C2R/reid/trainer.py` 中加入 AMP 混合精度训练（`autocast`）。
- 在 `Bi-C2R/reid/trainer.py` 中加入梯度累积机制（`grad_accum_steps`）。
- 在 `Bi-C2R/continual_train.py` 中加入 `GradScaler`，并打通 AMP/非 AMP 两条训练路径。
- 在训练入口新增参数：
  - `--amp` / `--no-amp`
  - `--grad-accum-steps`
- 默认参数调整为 16G 显存友好配置：
  - `batch-size=32`（原 64）
  - `workers=4`（原 8）
  - `amp=True`
  - `grad-accum-steps=2`
- 更新 `Bi-C2R/run1.sh`、`Bi-C2R/run2.sh` 与子项目 README 的推荐命令。
- 新增说明文档：`Bi-C2R/docs/LSTKCpp_Integration_16G_Design.md`。

### 这样做的原因

- 16G 显存是常见硬件上限，直接沿用大 batch 容易 OOM。
- 持续学习包含多损失联合优化，简单降 batch 会引入更大训练波动。
- AMP + 梯度累积可以在控制显存的同时，尽量保持等效 batch 统计稳定性。

### 使用建议

在 `Bi-C2R` 目录下优先使用以下命令：

```bash
bash run1.sh
# 或
bash run2.sh
```
