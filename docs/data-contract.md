# 资料与现有计算约定

## 修订快照

schema_version为整数1，metadata为业务方案字典，collections为业务集合字典。每条记录有非空字符串id和正整数revision。
同集合相同id取最高revision；同id同revision内容不同即冲突，即便存在更高修订也拒绝。deleted=true为撤回，之后更高修订可恢复。
撤回记录不要求携带业务字段。输入顺序与相同记录重试不改变语义；非有限数字拒绝。
Snapshot.rows返回独立副本；input_digest绑定有效修订和全部metadata。输出data_version沿用metadata，不由程序自造资料版本。
既有摘要入口为app.service.inspect(payload)，输出schema_version、input_digest、data_version和summary。

## 业务资料

months含month和values(变量键到数值或null)，metadata.variables给顺序，lags>=1。baseline.analyze仅用所有变量及前lags期均完整且月份连续的窗口生成Y和X，X先截距再从lag1到lagp按variables排。符号方案restrictions含shock、variable、horizon、sign，sign为+/-；B=P*Q，P为固定正对角Cholesky，Q正交，响应Psi_h*B；符号约束采用配置positive_margin，禁止每列另行挑不同Q。

结构分析沿用同一字段，并新增可选字段（均可缺省，缺省不改变旧接口）：
- metadata.shock_labels：长度等于变量数的唯一非空字符串数组。缺省按shock0..shockN-1。
  标签重复时拒绝，因为标签互换后的方案不可区分。
- metadata.restriction_version：字符串或整数，写在输出中便于按版本对照。
  也接受 metadata.scheme_version 作为别名。
- 同一(shock, variable, horizon)重复且同号的限制折叠为一条；符号相反即拒绝。
- 结构估计额外要求：保留样本月份严格连续、观测数大于回归元数、残差协方差正定。
  不满足时拒绝结构分析，但旧摘要入口仍按原语义工作。
- 结构分析不做变量变换。需要对数、差分或标准化时，应在资料准备阶段完成。

## 结构分析输出边界

app.structural.analyze 的 searched_feasible_ranges 是随机搜索命中的样本方案范围，
不是识别集合边界；certified_set 是区间证书，n=1、2 为解析精确边界，n=3 为
内/外近似（未决盒只进外近似），n>=4 不做边界声明。两者不可互相替代。
参数不确定性（自举约化式）与给定参数下的集合不确定性分开输出，不合并成百分位。
详细口径见 [结构分析方法](structural-method.md)。

各集合必要字段：{"months": ["month", "values"]}。

## 当前能力

app.baseline.analyze(snapshot)产生已有摘要，app.service.inspect负责绑定输入身份；
新增分析可以增设独立入口，已有接口及上述数据语义保持兼容。
结构分析入口为 app.structural.analyze(payload, options) 与
app.structural.compare_schemes(payload, entries, options)。
