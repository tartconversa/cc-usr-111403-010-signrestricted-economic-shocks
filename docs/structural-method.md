# 符号限制结构 VAR 方法与边界

本文说明新增结构分析的口径、可复核范围与不做什么。程序入口见
`app.structural.analyze` 与 `app.structural.compare_schemes`；数据读取仍走
`app.catalog` / `app.service`，旧摘要接口不变。

## 变量变换

程序**不做**差分、去均值或标准化。`docs/data-contract.md` 没有规定变换，
擅自加入会改变脉冲响应的解释。若业务上需要对数差分等变换，应在资料准备阶段
完成，再以所需变量键写入 `metadata.variables`。水平值 VAR 的响应按原变量单位
解释，且必须结合 `reduced_form.warnings`（近单位根、不稳定）一起读。

## 约化式

- 滞后窗口与 `app.baseline.analyze` 完全一致：`Y_t` 对常数与 `lag1..lagp` 的
  全部变量回归，`X` 顺序为 `[1, lag1(variables...), ..., lagp(variables...)]`。
- 额外要求：保留样本的月份严格连续。因为递归自举、历史分解与冲击的期数递推
  都要求无缺月；缺月直接拒绝，而不是静默插值。
- 诊断：伴随矩阵最大模特征值（稳定性）、设计矩阵条件数（共线）、残差协方差
  正定性与特征值、Ljung-Box 与 Jarque-Bera。后两者的尾概率用正态近似，只作
  提示，不宣称精确卡方检验。
- 残差协方差用自由度 `T - k` 归一化；`Sigma` 非正定时拒绝结构分解，而不是
  加抖动。

## 结构分解：一个 Q，全部限制

约定与资料契约一致：`B = P Q`，`P = chol(Sigma)` 为正对角固定因子，`Q` 为
正交矩阵；变量 `i` 对冲击 `j` 在期 `h` 的响应是 `[Phi_h B]_{i,j}`。

- **同一 Q**：一次分解只用一个 `Q`；全部冲击、变量、期数限制在同一个响应
  矩阵上同时检查，绝不按冲击分别挑列。
- `Q` 的取值域是完整的 `O(n)`，包含 `det=-1` 分支。只采样 `SO(n)` 会系统性
  漏掉“翻转单列后仍可行”的方案。
  搜索时把正态矩阵 QR 得到的 `SO(n)` 样本以等概率右乘固定反射
  `diag(-1,1,...)`，得到 `O(n)` 的 Haar 样本（`app/sign.py:haar_o`）。
- 符号限制要求 `sign * response >= positive_margin`，`margin` 是绝对下限，
  单位与数据相同。

## 正负号、标签与无限制冲击

- **符号翻转**：有限制列的符号由限制固定，翻转会破坏可行性，因此不会被当作
  新证据。无限制列在去重时取“绝对值最大元素为正”的规范方向；同一方案不会
  因整列变号被重复计数（`app/sign.py:canonicalize`）。
- **标签互换**：限制画像（同一冲击的全部限制集合）相同的冲击，其标签互换给出
  完全相同的可行集合。程序用 `label_equivalence_groups` 显式分组，搜索去重时
  对组内标签置换取规范序；范围汇总把同组冲击并成一组，标
  `label_not_distinguishable`，不分别报告“各自的”范围（否则会把同一方案当
  成两个独立证据）。
  用户应在 `metadata.shock_labels` 给出与顺序一致的唯一标签；标签重复直接
  拒绝。
- **无限制冲击**：其列只被正交性约束，在正交补内可取任意正交变换，因此**没有**
  可报告的逐期集合区间。搜索范围汇总与证书都标 `not_identified`，并明确不给
  数值范围。若把这类列与有限制的列并排报“范围”，等于把未识别的东西包装成
  识别结论，本实现拒绝这样做。

## 搜索 vs 证书

两者严格分开输出，字段名也不同：

- `searched_feasible_ranges`：Haar 随机搜索命中的可行方案汇总，标
  `kind: searched_range`。未命中时状态是 `no_feasible_sample_found`，注释写明
  “未找到不等于不可行”。搜索不参与任何矛盾判定。
- `certified_set`：区间证书。n=1、2 时参数域可精确参数化：
  - n=1：`Q ∈ {+1,-1}`，两点穷举即精确（`analytic_exhaustive`）；
  - n=2：响应是 `a sin t + b cos t`，可行集是有限段圆弧的精确并集，边界由
    反余弦给出（`analytic_arc_union`）；内=外，`gap` 为 0。
  - n=3：ZYZ 欧拉角参数化（`α,γ∈[0,2π)`, `β∈[0,π]`）加 `det=-1` 分支，用
    **均匀网格完整覆盖 + 区间二分精化**（`interval_uniform_cover_plus_bisection`）。
    每个盒用区间算术给出每条限制的上、下界：
    所有下界 `≥ margin` → 可行盒（内近似）；某条上界 `< margin` → 不可行盒；
    其余 → 未决盒。**未决盒只进外近似**。因此
    `inner ⊆ 真实识别集合 ⊆ outer`。
  - n≥4 不支持证书，只返回 `unsupported`，绝不猜测边界。
- 状态含义：
  - `certified_feasible_complete`：证书可行盒非空且未决测度为 0（n=1,2 或
    n=3 网格全部判定的情形）。
  - `certified_feasible_partial`：可行内近似非空，但仍有未决盒。此时**不能**
    宣称边界；只能报内/外区间。
  - `certified_infeasible`：完整覆盖下所有盒均不可行、且未决盒为 0。附
    `infeasibility_witness`，逐盒记录“最接近满足”的上界，并要求全部严格低于
    `margin`。随机搜索失败**不能**得出这个结论。
  - `inconclusive`：网格有未决盒但没有可行盒；既不能称可行也不能称不可行。
- 可复核性：`arithmetic` 字段写明区间运算的外扩（乘加 6 ULP、三角解析极值），
  `coverage_note` 写明覆盖与包含关系，`box_accounting` 给出种子盒数、
  可行/不可行/未决盒数，`feasible_boxes`/`undecided_boxes` 给出盒坐标与见证
  角度，可复算。这是数值证书，不是形式化证明。

## 参数不确定性 vs 集合不确定性

分开传播，绝不混成一组百分位：

- `parameter_uncertainty`：递归残差自举，每次抽样的**约化式**重新估计。
  `summary` 只报告约化式诊断的抽样分布（如最大特征值分位数）以及
  **每次抽样各自识别集合端点**的抽样分位数（`structural_set_endpoint_quantiles`）。
- `set_uncertainty`：给定点估计参数下的识别集合不确定，见证书的内/外近似。
- 程序不输出“某冲击某期响应的点识别 5%–95%”这类把不同抽样、不同旋转混在
  一起的百分位；`interpretation` 字段明确写出这一限制。

## 历史分解

`app/history.py:decompose` 用与当期分析**同一个可行 Q**：

    y_t = 初始条件_t + 确定项_t + Σ_s Σ_j (Phi_{t-s} B e_j) eps_{s,j}

- 初始条件 `S_{-1} = [y_{-1}; ...; y_{-p}]`（由首个保留行的 `X` 恢复），
  `A^{t+1} S_{-1}` 的顶部块进入第 `t` 期。
- 确定项 `Σ_k A^k d`，`d = [c; 0]`。
- 结构冲击 `eps_t = B^{-1} u_t`；程序同时输出结构冲击重构回约化残差的恒等式
  偏差 `reduced_residual_identity_gap`。
- 输出逐期重构值与 `reconstruction_error`，`checks.max_abs_reconstruction_error`
  用于检查恒等式；抽样测试中量级约 1e-15。
- 不会把不同 Q 的列拼进同一条历史路径；`structure_source` 记录该 Q 来自证书
  可行盒还是搜索命中。若两者都没有，则 `history_decomposition` 为空并给出
  `history_error`，而不是随便取一个 Cholesky 排序。

## 方案对照

`compare_schemes(payload, entries)` 每个方案各自搜索、各自证书，再按
`(shock, variable, horizon)` 并列展示：

- 每个方案带 `fingerprint`（变量、margin、限制内容的哈希）与可选
  `restriction_version`，便于按版本复核。
- 只有变量顺序与 `shock_labels` 完全一致时才逐冲击比较；不一致直接拒绝，
  避免把不同标签的数值并排当证据。
- 对照表中同时给出 `searched_found_min/max`（搜索）与
  `certified_inner/outer`（证书），二者不合并。

## 命令行

    python -m app.structural 资料.json [配置.json]

配置是 JSON 对象，`horizons`、`search_attempts`、`search_target`、`seed`、
`certify_refine_budget`、`bootstrap_draws`、`bootstrap_certify` 等见
`app/structural.py:DEFAULT_OPTIONS`。若配置含 `compare_entries`，入口转为
方案对照。

## 已知局限

- 结构分析不含变量变换、不含滞后阶数选择搜索；`lags` 由资料给出，程序只报
  诊断。
- n≥4 无证书，只提供搜索范围，且明确不是边界。
- n=3 证书的内近似可能偏小（区间算术保守）；要缩窄需增大
  `certify_refine_budget` 或网格，或接受 `certified_feasible_partial` 的外近似
  只作保守约束。若业务需要精确边界，应回到 n≤2 或改用可精确求解的限制方案。
- 数值证书假定 `Sigma` 与 `beta` 为准；参数不确定性的传播见上一节，二者不
  合并。
