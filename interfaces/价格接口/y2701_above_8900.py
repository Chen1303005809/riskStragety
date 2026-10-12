from external_head import *


class My_TEST(RHTemplate):
    """Notify once when the latest y2701 price rises above 8900."""

    paramMap = {}
    varMap = {}

    def __init__(self, pid) -> None:
        self.symbolList = ["y2701"]
        self.exchangeList = ["DCE"]
        self.npid = pid
        self.notification_sent = False

        super().__init__()

        self.instrument = "y2701"
        self.exchange = "DCE"

    def onStart(self) -> None:
        super().onStart()

    def onStop(self) -> None:
        super().onStop()

    def onTick(self, tick: VtTickData) -> None:
        super().onTick(tick)

        if (
            not self.notification_sent
            and tick.vtSymbol == "y2701"
            and tick.lastPrice > 8900
        ):
            self.putEvent(
                "y2701 latest price {:.1f} is above 8900".format(
                    tick.lastPrice
                ),
                "warning"
            )
            self.notification_sent = True
