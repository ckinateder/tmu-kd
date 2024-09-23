from datetime import datetime, timedelta

def add_time(eta_seconds: int) -> datetime:
    """Increment the current time by the given number of seconds.

    Args:
        eta_seconds (int): seconds to add

    Returns:
        datetime: datetime object
    """
    now = datetime.now()
    completion_time = now + timedelta(seconds=eta_seconds)
    return completion_time