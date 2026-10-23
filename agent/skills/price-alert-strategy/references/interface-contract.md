# 价格接口契约参考

本参考根据 `interfaces/价格接口/external_head.py` 和 `external_baseid.py` 整理。它描述运行接口，不包含策略业务需求。修改底层接口后应同步检查和更新本参考。

## 导入与行情

- 使用 `from external_head import RHTemplate, VtTickData`。不要直接导入 `PyEngine` 或 `external_baseid`。
- `VtTickData` 提供 `vtSymbol`、`lastPrice`、`lastVolume`、`bidPrice1` 和 `askPrice1`。本项目的策略只需要 `vtSymbol` 与 `lastPrice`；不要依赖行情时间戳，当前类型没有该字段。
- `RHTemplate.onTick(tick)` 会把 tick 数据复制到基类的 `on_tick_data`。对有效的 `VtTickData`，策略应调用一次 `super().onTick(tick)`，再用原始 `vtSymbol` 和有效正数 `lastPrice` 处理行情。

## 策略初始化与订阅

- 策略顶层类名固定为 `My_PythonObj`，直接继承 `RHTemplate`；类属性 `name` 是文件名依据，必须为英文 ASCII 标识。工具在保存时会据此生成 `<name>.py`。
- 初始化 `RHTemplate` 前设置 `self.npid`、字符串列表 `self.symbolList`、与之等长且逐项对应的 `self.exchangeList`，以及策略状态。随后只调用一次 `super().__init__()`。
- `RHTemplate.__init__()` 不会初始化 `vtSymbol`。父类初始化后设置 `self.vtSymbol = ";".join(self.symbolList)`、`self.exchange = ";".join(self.exchangeList)`、`self.instrument = self.vtSymbol`。
- 基类 `onStart()` 调用 `subInstrument()`，后者逐项订阅 `self.symbolList`。不要重写订阅逻辑，不要把交易所拼进合约代码。平台负责调用生命周期方法和 `setParam()`；不要猜测或伪造 `sid`。
- `paramMap` 仅用于平台界面参数映射；无明确界面参数需求时保持 `{}`，将用户指定的策略参数放在模块级常量中。基类初始化会把实例 `varMap` 设为空字典。

## 日志与通知

- 用 `self.output(content)` 写策略日志；不要使用 `print`、`logging` 或直接调用引擎接口。
- 用 `self.putEvent(message, level)` 发送通知，价格预警级别为 `"warning"`。不得调用 `buy`、`short` 或其他下单接口。
- 基类 `onStart()` 会记录启动并订阅，`onStop()` 会记录停止；策略应调用对应父类方法。不要在模块导入时创建策略实例或手动启动策略。

## 运行与沙盒边界

- `RHTemplate` 的公开 API 包含 `onStart()`、`onStop()`、`onTick()`、`output()` 和 `putEvent()`。基类内部使用 `PyEngine` 与 `external_baseid` 常量；策略不应绕过公开 API 调用它们。
- 功能测试沙盒提供 `PyEngine` 与 `RHclient` 的模拟对象，不联网、不连接真实行情引擎，也不验证真实引擎的部署行为。
