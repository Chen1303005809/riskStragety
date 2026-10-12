# encoding: UTF-8
import PyEngine as RHEngine
from external_baseid import*
import json
import os

from typing import Any, Dict, List
from dataclasses import dataclass
from collections import OrderedDict, defaultdict

#交易方向类型
Client_TRADE_D_Buy = '0'
Client_TRADE_D_Sell = '1'

@dataclass
class VtTickData():
    """Tick 行情数据类"""
    vtSymbol = "" #合约代码
    
    lastPrice = 0.0 #最新成交价
    lastVolume = 0 #最新成交量

    bidPrice1 = 0.0
    askPrice1 = 0.0

@dataclass
class TickData(VtTickData):
    """带最新成交量的最新 Tick 数据"""
    _cache_volume = 0 # 缓存总成交量

    @property
    def last_volume(self) -> int:
        last_volume: int = self.volume - self._cache_volume

        if not self._cache_volume:
            last_volume = 0
        
        self._cache_volume = self.lastVolume
        return last_volume

    def update(self,tick: "TickData") -> None:
        self.__dict__.update(tick.__dict__)

class RHTemplate(object):
    name = '' #策略名称

    def __init__(self) -> None:
        self.base_param_list = [
            'name',
            'investor',
            'instrument',
            'exchange'
        ]
        self.base_var_list = [
            'trading',
            'pos'
        ]

        # self.paramMap: Dict[str,str] = {}
        self.varMap: Dict[str,str] = {}

        self.instrument = '' #合约
        self.exchange = '' #交易所代码
        self.investor = ''
        #self.volume = 0
        self.trading = False #是否启动策略
        self.pos: Dict[str,int] = {} #总投机持仓
        self.longPrice = 0.0 #多头开仓价
        #self.symbolList = [] #所有需要订阅的合约
        self.clientEngine = RHEngine.PyEngine()
        self.on_tick_data = TickData()
        

    @property
    def paramList(self) -> List[str]:
        return self.base_param_list + list(self.paramMap.keys())
    
    @property
    def varList(self) -> List[str]:
        return self.base_var_list + list(self.varMap.keys())
    
    @property
    def className(self) -> str:
        return self.__class__.__name__
    
    #todo 参数保存文件

    def subInstrument(self):
        for instrument in self.symbolList:
            instrument = instrument.strip()
            if not instrument:
                continue

            req = {
                'MsgType': PythonGoSubscribeInstrument,
                'pid': self.npid,
                'pyPid': os.getpid(),
                'instrument': instrument
            }

            json_str1 = json.dumps(req)
            self.clientEngine.sendMsg(json_str1, "")

            self.output(
                "subscribe request sent: {}".format(instrument)
            )
    
    def output(self, content: str) -> None:
        """打日志"""
        req = {
            'Msg' : content,
            'pid' : self.npid,
        }
        req['MsgType'] = PyToSpiLog
        json_str1 = json.dumps(req)
        json_str2 = ""
        self.clientEngine.sendMsg(json_str1,json_str2)

    def putEvent(
        self,
        message: str = "strategy parameters updated",
        level: str = "info"
    ) -> None:
        """发送策略通知"""
        req = {
            "MsgType": PyToSpiNotification,
            "pid": self.npid,
            "level": level,
            "message": message,
        }
        self.clientEngine.sendMsg(json.dumps(req), "")

    def onStart(self) -> None:
        """启动策略"""
        self.trading = True
        self.output("stategy started already")
        self.subInstrument()
    
    def onStop(self) -> None:
        """停止策略"""
        self.trading = False

        self.output("stategy stopped")
    
    def onTick(self, tick: VtTickData) -> None:
        """收到行情TICK推送"""
        self.on_tick_data.update(tick)
    
    def buy(self,price,volume,instrument, exchange, memo = None, investor = ''):
        """买开"""
        exchange = exchange
        req = {
            'sid' : 0,
            'hedflag' : 1,
        }
        req['direction'] = '0'
        req['volume'] = volume
        req['orderPrice'] = price
        req['instrument'] = instrument
        req['pid'] = self.npid
        req['MsgType'] = PythonSendOrder
        #todo 向客户端发送报单请求 
        json_str1 = json.dumps(req)
        json_str2 = ""
        self.clientEngine.sendMsg(json_str1,json_str2)
    
    def short(self,price,volume,instrument, exchange, memo = None, investor = ''):
        """卖开"""
        # instrument = instrument or self.symbolList[0]
        exchange = exchange
        req = {
            'sid' : 0,
            'hedflag' : 1,
        }
        req['direction'] = '1'
        req['volume'] = volume
        req['price'] = price
        req['instrument'] = instrument
        req['pid'] = self.npid
        #todo 向客户端发送报单请求 
        json_str1 = json.dumps(req)
        json_str2 = ""
        self.clientEngine.sendMsg(json_str1,json_str2)

    def setCLientPid(self,pid):
        self.npid = pid
    
    def Response(self):
        return self.spi
    
    def setParam(self, setting: dict):
        aa = list(setting.keys())
        bb = ','.join(aa)
        self.output(bb)
        """刷新参数, 修改界面参数时调用"""
        params = {"sid": self.sid}
        map_keys = list(self.paramMap.keys())
        map_values = list(self.paramMap.values())

        for key, value in setting.items():
            #: 更新类属性, 并把更新后的数据回传给客户端
            params[key.encode('gbk')] = setting[key]
            class_attr_name = map_keys[map_values.index(key)]
            if class_attr_name != 'vtSymbol':
                #: 证券代码是纯数字, 不能转成整型
                value = eval(value)
            setattr(self, class_attr_name, value)

        # 初始化仓位信息
        self.symbolList = self.vtSymbol.split(';')
        self.exchangeList = self.exchange.split(';')

        self.pos: Dict[str, int] = {}
        self.tpos0L: Dict[str, int] = {}
        self.tpos0S: Dict[str, int] = {}
        self.ypos0L: Dict[str, int] = {}
        self.ypos0S: Dict[str, int] = {}

        for symbol in self.symbolList:
            if not symbol: continue
            self.pos[symbol] = 0
            self.ypos0L[symbol] = 0
            self.tpos0L[symbol] = 0
            self.ypos0S[symbol] = 0
            self.tpos0S[symbol] = 0

        self.putEvent()
    
    def getParam(self):
        """获取参数"""
        return list(self.paramMap.values())
        setting = OrderedDict()
        for key in reversed(self.paramList):
            if key in self.paramMap:
                setting[self.paramMap[key]] = str(getattr(self, key))
        return setting
    
    
    
        


# class Head:
#     def __init__(self) -> None:
#         self.spi = None
        
    
#     def SetSpi(self,spi):
#         self.spi = spi
    
#     def Response(self,arg):
#         return self.spi(arg)
    
#     def subIns(self,arg):
#         apyPid = os.getpid()
#         hh = RHEngine.SubInstrument(ExchangeID = "", InstrumentID = "cu2311", nClientPid = arg, nPyPid = apyPid)
#         hh.subIns()
    
# head = Head()

# def GetHead():
#     return head

# def Response(arg):
#     head.Response(arg)

# def subsub(arg):
#     apyPid = os.getpid()
#     hh = RHEngine.SubInstrument(ExchangeID = "", InstrumentID = "IF2309", nClientPid = arg, nPyPid = apyPid)
#     hh.subIns()
