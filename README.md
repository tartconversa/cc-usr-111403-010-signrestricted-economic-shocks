# 工业外贸价格冲击的符号识别集合与动态响应

本项目在现有资料修订读取与基础统计之上，增加带符号限制的结构 VAR 分析：
同一正交旋转下同时检验全部冲击限制，输出“搜索到的可行范围”与“认证识别集合
边界”两类不同性质的结果，并分离参数不确定性与集合不确定性。

app/catalog.py 按集合和记录键解析修订，拒绝同修订冲突；app/baseline.py 承载现有
业务摘要；app/service.py 输出摘要及输入摘要哈希。
结构分析新增于 app/reduced.py（约化式与诊断）、app/sign.py（限制编译、旋转搜索、
等价类）、app/certify.py（区间证书）、app/history.py（历史分解）、app/structural.py
（编排与方案对照）。
数据字段及既有接口见 [资料契约](docs/data-contract.md)；结构分析的方法口径与
可复核范围见 [结构分析方法](docs/structural-method.md)。
fixtures/sample.json 是合成小数据，不对应真实企业或个人；样本太小，结构分析会
明确拒绝估计，而不会给出不可靠结果。
运行环境为 Python 3.14，全部模块仅使用标准库。

读取并计算当前摘要：

    python -m app.service fixtures/sample.json

运行结构分析（第二个参数可选，为配置 JSON）：

    python -m app.structural fixtures/sample.json
    python -m app.structural data.json config.json

运行全部模块检查：

    python -m unittest discover -s tests -v

检查源码编译：

    python -m compileall -q app tests
