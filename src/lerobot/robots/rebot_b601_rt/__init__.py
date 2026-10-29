from .config_rebot_b601_rt import RebotB601RTConfig


def __getattr__(name):
    # Register configuration without loading native libraries for other robots.
    if name == "RebotB601RT":
        from .rebot_b601_rt import RebotB601RT

        return RebotB601RT
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
