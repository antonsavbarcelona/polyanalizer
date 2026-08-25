"""Signal-response discovery pipeline.

    raw collector data -> features -> SignalConfig -> Signal -> EntryMark
    -> SignalResponse -> PathStats -> Controls -> Aggregation
    -> SignalConfigSummary -> Plateau/robustness analysis

FORBIDDEN anywhere in this package: TP, SL, timeout exit, trade winrate,
exit-policy optimization, portfolio simulation. The only question this
pipeline answers is: what happened to the executable Polymarket price after
a potential Binance signal? See IMPLEMENTATION CONTRACT section 1.
"""
