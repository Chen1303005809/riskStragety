# PythonGo K 线接口契约参考

本参考提炼自用户提供的《PythonGo-K线接口接入说明.md》，作为 Agent 生成策略时的接口资料。附件中的合约、价格、日期和完整代码仅为说明示例，不是业务需求或默认值。此参考不证明当前部署的 DLL、Python 支撑模块或行情服务已更新。

## 依赖与策略生命周期

- 策略需要由宿主提供带 K 线能力的 `external_head.py`，其中包含 `RHTemplate`、`VtBarData` 和下述方法；宿主还需要匹配的 `external_baseid.py`、PythonGo/Dispatch 桥接和可用 K 线服务。
- 使用 `from external_head import RHTemplate, VtBarData`。策略不得直接导入 `PyEngine` 或消息号常量。
- 策略类名固定为 `My_PythonObj`。遵循本项目的 `RHTemplate` 初始化和文件产物约定；只用 K 线时 `symbolList`、`exchangeList` 可为空，基类启动时不会替代 K 线订阅。
- 宿主先设置策略上下文，再调用策略生命周期。不要在构造函数中查询 K 线，也不要硬编码策略身份。
- `super().onStart()` 会按 `symbolList` 订阅 Tick；`super().onStop()` 清理 Python 侧 pending 请求与订阅状态。只手工调用对象的 `onStop()` 不等同于宿主停止流程。

## 周期枚举

`cycle_type` 必须是 1 到 14 的枚举值：

| 值 | 周期 |
| --- | --- |
| 1 | 1 分钟 |
| 2 | 1 小时 |
| 3 | 日线 |
| 4 | 周线 |
| 5 | 月线 |
| 6 | 5 分钟 |
| 7 | 3 分钟 |
| 8 | 6 分钟 |
| 9 | 10 分钟 |
| 10 | 15 分钟 |
| 11 | 30 分钟 |
| 12 | 季线 |
| 13 | 半年线 |
| 14 | 年线 |

枚举值不是分钟数。查询区间决定取数范围，`cycle_type` 决定每根 K 线周期，两者相互独立。

## 历史查询

```python
queryHistoryKline(
    instrument: str,
    cycle_type: int,
    count: int | None = None,
    exchange: str = "",
    product: str = "",
    start_date: str = "",
    start_time: str = "",
    end_date: str = "",
    end_time: str = "",
    market_type: int = 1,
    global_id: int = 0,
) -> int
```

- `instrument` 是非空 ASCII 合约代码且小于 61 字节；`exchange` 是 ASCII 且小于 9 字节；`product` 是 ASCII 且小于 61 字节。建议显式提供交易所和品种。
- `count` 省略时默认 500。查询区间时提供完整的开始、结束日期时间，`count` 为 0 或省略；最新 N 根时不提供日期时间，`count` 为 1–2000；截至结束时间的 N 根时只提供完整结束日期时间，`count` 为 1–2000。
- 开始和结束日期、时间必须各自成对。日期格式为 `YYYYMMDD`，时间格式为 `HH:MM:SS`，各字段是 ASCII 且小于 9 字节。接口只做有限格式与顺序检查，策略仍需提供真实有效的用户日期。
- `market_type` 当前只允许 1。普通场景使用 `global_id=0`；其他值依赖具体服务配置。
- 返回值是 request ID，不是 bars。请求提交前的参数或发送异常需在调用处处理；未成功提交的请求不会通过历史回调报告错误。

### 历史回调

```python
onHistoryKline(request_id: int, bars: list[VtBarData], error: dict | None) -> None
```

- 桥接层可能接收多包后聚合，再调用策略一次。策略不处理底层包结束标志，不重写内部 `_onHistoryKlineComplete`。
- `error is None` 且 `bars == []` 表示正常空结果。错误时 `bars` 可能有部分数据，必须先检查错误并丢弃这些部分结果。
- 回包可能乱序；多请求按 request ID 关联。桥接层按收到顺序追加，没有统一排序或去重。
- 错误码包括 `KLINE_RESPONSE_OVERFLOW`、`KLINE_QUERY_CANCELLED`、`KLINE_API_UNAVAILABLE`、`STRATEGY_NOT_RUNNING`、`KLINE_SUBMIT_FAILED`、`KLINE_QUERY_TIMEOUT` 和通用 `KLINE_QUERY_FAILED`。错误对象包含 `code`、`errorId`、`message`、`requestId`。

## 实时订阅与退订

```python
subscribeKline(
    instrument: str,
    cycle_type: int,
    exchange: str = "",
    product: str = "",
    market_type: int = 1,
    global_id: int = 0,
) -> None

onKline(bar: VtBarData) -> None

unsubscribeKline(
    instrument: str,
    cycle_type: int,
    exchange: str = "",
    product: str = "",
    market_type: int = 1,
    global_id: int = 0,
) -> None
```

- 历史查询不自动订阅实时数据，实时订阅也不自动补历史；用户要求时才组合。
- `onKline` 每次收到一个 bar，不带历史 request ID。多合约或周期按 `vtSymbol`、`exchange`、`cycleType` 等字段区分。
- bar 没有 `is_closed` 或收盘确认标志。`closePrice` 字段名不代表实时推送已收盘；推送频率、时间字段起止含义和同一根重复更新行为需以实际服务确认。
- 退订使用与订阅相同的参数。宿主订阅键为 `(market_type, instrument, cycle_type, exchange, product, global_id)`，共享订阅由宿主管理所有者。
- 订阅和退订没有独立 ACK。调用没有异常只说明本地校验与发送过程未抛异常；已排队事件也不保证在停止或退订后立即撤销。
- 历史与实时衔接不保证无数据缺口。若需缓存合并，候选 bar 键为 `(exchange, vtSymbol, cycleType, tradeDate, time, globalId)`，但其可靠性取决于服务的时间标识稳定性，须实际验证。

## `VtBarData` 字段

历史结果列表和实时回调使用相同对象：

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `vtSymbol` | `str` | 合约代码，不拼交易所后缀 |
| `exchange` | `str` | 交易所代码 |
| `cycleType` | `int` | 周期枚举 |
| `tradingDay` | `str` | 交易日 |
| `tradeDate` | `str` | 自然日 |
| `time` | `str` | 服务返回的时间标记 |
| `openPrice` / `highPrice` / `lowPrice` / `closePrice` | `float` | OHLC 价格；实时 `closePrice` 不代表收盘确认 |
| `openInterest` | `float` | 持仓量 |
| `volume` | `float` | 成交量，累计或单根口径由服务确定 |
| `volumeDelta` | `float` | 成交量变动，口径由服务确定 |
| `turnover` | `float` | 成交额，单位由服务确定 |
| `globalId` | `int` | 底层全局标识 |

桥接直接映射这些字段，不做复权、时区转换、交易日转换或成交量口径转换。尤其夜盘中不可混淆 `tradingDay` 和 `tradeDate`。
