from external_head import *
# import sys
# import numpy as np
# import os  

class My_TEST(RHTemplate):
    """小航不会写代码"""
    # 参数映射表
    paramMap = {
        'exchange':'交易所',
        'vtSymbol':'合约',
        'volume':'委托数量',
        'price':'触发价格'
    }
    # 参数英文字段
    paramList = list(paramMap.keys())
    # 变量映射表
    varMap = {
        'trading': '交易中',
        'pos': '持仓'
    }
    # 变量英文字段
    varList = list(varMap.keys())
    
    def __init__(self,pid: Any) -> None:
        self.symbolList = ["ag2610","ag2611"]
        self.exchangeList = ["SHFE","SHFE"]
        self.volume = 10
        self.npid = pid
        self.price = 1 #此价格为策略两合约的差价
        super().__init__()
        self.exchange = '' # 交易所和合约从界面填入，所以留空
        self.vtSymbol = ''
        self.order_count = 3 # 报单次数
        self.tick_price1 = 0
        self.tick_price2 = 0
        self.receivedTickSymbols = set()
        

    def onTick(self, tick: VtTickData) -> None:
        """收到行情 Tick，仅用于验证，不执行报单"""
        super().onTick(tick)

        # 每个合约只记录第一笔行情，避免高频日志淹没页面。
        if tick.vtSymbol not in self.receivedTickSymbols:
            self.receivedTickSymbols.add(tick.vtSymbol)

            super().output(
                "onTick received: instrument={}, lastPrice={}".format(
                    tick.vtSymbol,
                    tick.lastPrice
                )
            )


    def onStart(self) -> None:
        """客户端点击运行按钮回调"""
        super().onStart() # 调用父类的 onStart 方法

        self.putEvent("notification pipeline test", "info")
    
    def onStop(self) -> None:
        """客户端点击暂停按钮回调"""
        super().onStop() # 调用父类的 onStop 方法

    # def onTrade(self, trade: VtTradeData) -> None:
    #     """成交回报回调"""
    #     super().onTrade(trade, log=True) # 调用父类的 onTrade 方法
    #     self.putEvent() # 成交后更新界面显示的持仓数据
    #     # 你可以在这里写成交后策略的运行逻辑

# 测试代码
# instance = My_TEST(118237)
# instance.onStart()
# a = VtTickData()
# a.lastPrice = 66
# a.vtSymbol = "zhuzhu"
# instance.onTick(a)
