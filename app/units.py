def bytes_to_mb(value: int) -> int:
    """Bytes -> MB (base 1000). Manufacturers' 'on the box' convention."""
    return value // (10**6)


def bytes_to_mib(value: int) -> int:
    """Bytes -> MiB (base 1024). The convention GPU tools display."""
    return value // (2**20)


def format_large_numbers(large_num: int) -> str:
    return f"{large_num:,}".replace(",", " ")
