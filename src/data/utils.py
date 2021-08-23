from pathlib import Path


def represents_int(s: str):
    try:
        int(s)
        return True
    except ValueError:
        return False


def get_data_size(path: Path):
    data_size = 0
    for entry in path.iterdir():
        if represents_int(entry.name.removesuffix(entry.suffix)):
            data_size += 1
    return data_size
