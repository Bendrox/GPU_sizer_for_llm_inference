def oct_to_mo(input: int) -> int:
    """Octets -> Mo (base 1000). Convention 'sur la boîte' des fabricants."""
    return input // (10**6)


def oct_to_mio(input: int) -> int:
    """Octets -> Mio (base 1024). Convention 'à l'affichage' des outils GPU."""
    return input // (2**20)


def format_large_numbers(large_num: int) -> str:
    return f"{large_num:,}".replace(",", " ")