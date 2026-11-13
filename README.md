# 工业外贸价格冲击的符号识别集合与动态响应

本项目提供约化VAR稳定性及估计域相关资料的现有业务摘要。当前代码执行资料修订读取与基础统计，不包含扩展分析。

app/catalog.py 按集合和记录键解析修订，拒绝同修订冲突；app/baseline.py 承载当前业务计算；app/service.py 输出摘要及输入摘要哈希。
数据字段及现有接口见 [资料契约](docs/data-contract.md)。fixtures/sample.json 是合成小数据，不对应真实企业或个人。
运行环境为 Python 3.14，现有模块仅使用标准库。

读取并计算当前摘要：

    python -m app.service fixtures/sample.json

运行当前模块检查：

    python -m unittest discover -s tests -v

检查源码编译：

    python -m compileall -q app tests
