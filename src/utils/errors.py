def require(condition, message="", error=ValueError):
    if not condition:
        raise error(message)
