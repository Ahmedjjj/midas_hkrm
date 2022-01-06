def require(condition: bool, message: str = "", error: Exception = ValueError):
    """

    Helper function for checking errors

    Args:
        condition (bool): condition that should be true.
        message (str, optional): Error message. Defaults to "".
        error (Exception, optional): Exception to throw. Defaults to ValueError.

    Raises:
        error: [description]
    """
    if not condition:
        raise error(message)
