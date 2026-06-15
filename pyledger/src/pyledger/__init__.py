# from .simple import access
from .simple import access, access2, access3

def hello() -> str:
    return "Hello from pyledger!"

__all__ = ["hello", "access", "access2", "access3"]
