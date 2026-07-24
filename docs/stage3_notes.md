# Stage 3 Baseline Notes

## 1. Stage 3 目标

第三阶段在 ROI 内检测竖直基准线两侧相邻的黑色条纹中心线：

- 检测基准线左侧相邻的一条黑色条纹中心线。
- 检测基准线右侧相邻的一条黑色条纹中心线。
- 分别输出 `center_x_roi`、`center_x_global` 和 `distance_px`。
- 只有左右两侧都找到有效条纹时，整体结果才为 `success = true`。

距离定义为非负的水平像素距离：

- Left：`x_ref_roi - left_center_x_roi`
- Right：`right_center_x_roi - x_ref_roi`

左右轨迹独立统计有效行。跨越基准线的黑色 run 可以参与候选，但整条轨迹最终根据 median center 位于基准线左侧还是右侧进行分侧。同一轨迹不会同时作为左右结果。

## 2. 当前 Baseline 算法流程

### 2.1 输入 mask

第三阶段使用 `black_mask.png`：

- 白色像素代表原始灰度图中的黑色区域。
- 黑色像素代表原始灰度图中的白色条纹。

### 2.2 Row-wise black run extraction

逐行扫描 `black_mask`，提取每个白色连续区间 `[x0, x1)`，并计算：

- `width_px`
- `center_x_roi = (x0 + x1 - 1) / 2`

偶数宽度区间的中心可以是半像素坐标，例如 `300.5`。

### 2.3 Run prefilter

每个 run 先按以下条件初筛：

- 是否接触 ROI 左右边界。
- 宽度是否小于 `min_run_width_px`。
- 宽度是否大于 `max_run_width_px`。
- 中心是否超出基准线搜索半径。

当前 Stage 3 参数为：

| 参数 | 当前值 |
| --- | ---: |
| `stripe_search_radius_px` | 120 |
| `min_run_width_px` | 3 |
| `max_run_width_px` | 120 |
| `center_cluster_tolerance_px` | 5 |
| `max_width_deviation_ratio` | 0.75 |
| `min_width_tolerance_px` | 3 |
| `min_stripe_support_ratio` | 0.5 |
| `reject_border_touching_runs` | `true` |

### 2.4 Center voting / clustering

每个通过初筛的 run 按自己的中心位置贡献一票。算法沿 x 方向对中心票数进行窗口聚合，形成近似竖直的条纹候选 seed。

一个宽 run 仍然只贡献一票，不会因为面积更大而按像素数量获得额外权重。

### 2.5 Track-level filtering

每个 seed 在每一行最多分配一个中心距离最近的 run。随后根据该轨迹的初始 center median 和 width median 进行第二次过滤：

- 中心偏差过大：`center_only`
- 宽度偏差过大：`width_only`
- 中心和宽度都异常：`center_and_width`

`assigned rows` 是进入轨迹二次判断的行数；`valid rows` 是过滤后真正用于最终中心计算的行数。

```text
retention ratio = valid rows / assigned rows
valid row ratio = valid rows / ROI height
```

### 2.6 左右分侧与最终选择

轨迹使用有效 run 的 median center 作为最终 `center_x_roi`：

- median center 小于基准线：left candidate
- median center 大于基准线：right candidate
- median center 等于基准线：不属于任意一侧

算法分别选择距离基准线最近的 eligible left 和 eligible right。任意一侧缺失时，整体结果失败，不使用宽松 fallback 生成结果。

## 3. Debug 输出解释

### `black_run_candidates.png`

- 绿色点：通过初筛的 run 中心；每个点代表一票。
- 暗红色点：初筛被拒绝的 run 中心。
- 橙色 run：已分配给 selected track，但在 track-level filtering 中被拒绝。
- 蓝色 run：selected left track 的最终有效 run。
- 黄色 run：selected right track 的最终有效 run。
- 洋红色竖线：ROI 内基准线。

这张图用于区分“通过初筛并参与投票”和“真正进入最终轨迹”的 run。

### `stripe_center_votes.png`

- 绿色曲线表示沿 x 方向聚合后的中心票数。
- 红色竖线表示基准线。
- 蓝色标记表示 selected left track。
- 黄色标记表示 selected right track。
- 白色标记表示 eligible 但离基准线更远的候选。
- 灰色标记表示不满足支持率等条件的候选。

图中同时显示 selected tracks 的 seed、`assigned -> valid` 和 retention。

### `adjacent_stripes_result.png`

- 红色竖线：基准线。
- 蓝色竖线和点：最终 left centerline 及其有效 run 中心。
- 黄色竖线和点：最终 right centerline 及其有效 run 中心。
- 顶部文字显示最终 x、距离和 `assigned -> valid`。

### `stripe_results.json`

JSON 记录：

- 当前算法参数和 ROI/global 基准线。
- 初筛接受与各类拒绝数量。
- 每个 candidate 的 seed 票数、初始 median、宽度容差和最终 median。
- `assigned_row_count`、`valid_row_count`、`valid_row_ratio` 和 `retention_ratio`。
- `center_only`、`width_only`、`center_and_width` 拒绝计数。
- crossing run 的 assigned/valid 数量。
- `eligible`、`selected`、`selection_reason` 和 `rejection_reasons`。
- 最终左右中心、距离及整体 success 状态。

## 4. Sample 1 验证结论

基准线为 `x_ref_roi = 300`、`x_ref_global = 1000`。

| 指标 | Left | Right |
| --- | ---: | ---: |
| `center_x_roi` | 286.0 | 300.5 |
| `center_x_global` | 986.0 | 1000.5 |
| `distance_px` | 14.0 | 0.5 |
| Assigned rows | 677 | 882 |
| Valid rows | 677 | 641 |
| Retention ratio | 1.0 | 0.727 |
| Median width | 9 px | 10 px |

Right track 的二次拒绝：

```text
center_only: 0
width_only: 161
center_and_width: 80
```

Sample 1 中央大面积黑洞确实通过了初筛并参与中心投票，也进入了 selected right track 的初始 assigned rows。但是异常部分在 track-level filtering 中因宽度或中心与宽度同时异常而被拒绝。

最终 right track 的有效 run：

- 宽度中位数为 10 px。
- 宽度 P90 为 11 px。
- 最大宽度为 12 px。

因此中央黑洞没有成为最终有效轨迹的主体。最终左右中心线肉眼可接受，Sample 1 通过当前 baseline 验证。

## 5. Sample 2 验证结论

基准线为 `x_ref_roi = 300`、`x_ref_global = 1000`。

| 指标 | Left | Right |
| --- | ---: | ---: |
| `center_x_roi` | 267.5 | 307.0 |
| `center_x_global` | 967.5 | 1007.0 |
| `distance_px` | 32.5 | 7.0 |
| Assigned rows | 974 | 784 |
| Valid rows | 974 | 784 |
| Retention ratio | 1.0 | 1.0 |
| Valid row ratio | 0.974 | 0.784 |
| Median width | 27 px | 44 px |
| Width P90 | 30 px | 47 px |
| Max width | 30 px | 49 px |

Sample 2 selected left/right 均没有 track-level rejection：

```text
center_only: 0
width_only: 0
center_and_width: 0
```

Right track 的 784 个 assigned runs 全部跨越基准线，也全部通过二次过滤。该轨迹的 median center 为 `307.0`，因此按当前项目定义归入右侧。

Sample 2 的 right stripe 比 Sample 1 正常条纹更宽，但宽度稳定且近似竖直。其 selected tracks 没有受到大黑洞候选污染。与 Sample 1 相比，Sample 2 的投票和轨迹结构更干净，Sample 2 通过当前 baseline 验证。

## 6. 当前不建议改参数的理由

- Sample 1 包含窄黑色条纹和中央黑洞，当前 track-level filtering 能排除异常宽 run。
- Sample 2 包含正常但明显更宽的 right stripe，宽度中位数为 44 px、最大宽度为 49 px。
- 现在收紧 `max_run_width_px` 可能破坏 Sample 2 或未来其他正常宽条纹。
- 当前只有两个真实样本，依据这两个样本调参容易过拟合。

当前参数同时覆盖了 Sample 1 的窄条纹、Sample 1 的黑洞干扰和 Sample 2 的较宽正常条纹，因此暂时保持不变。

## 7. 当前明确不做

当前 baseline 不包含：

- UI
- Automatic ROI
- Hough line detection
- Skeletonize
- Deep learning
- Broad fallback 或复杂自动回退策略

## 8. 后续风险

- 当前只有 Sample 1 和 Sample 2，不能据此证明算法具有泛化能力。
- 稳定、狭长且支持行数较高的异常黑区，未来仍可能表现得像真实条纹。
- ROI、预处理 mask 或真实条纹宽度发生明显变化时，当前参数可能不再适用。
- 后续需要使用更多真实样本验证窄条纹、宽条纹、黑洞、粘连和缺失条纹等情况。
- 如果未来出现误判，应先读取 `stripe_results.json` 的 assigned/valid、retention、宽度和 rejection diagnostics，再基于实际证据调整过滤或参数。

当前结论是：第三阶段可作为可解释、可 debug 的初始 baseline，但仍需要更多数据验证，不能将两个样本通过等同于算法已经泛化。
