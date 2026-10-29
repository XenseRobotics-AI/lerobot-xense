from .config_bi_rebot_b601_rt import BiRebotB601RTConfig


def __getattr__(name):
    # Defer native drivers until this robot is actually selected.
    if name == "BiRebotB601RT":
        from .bi_rebot_b601_rt import BiRebotB601RT

        return BiRebotB601RT
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
