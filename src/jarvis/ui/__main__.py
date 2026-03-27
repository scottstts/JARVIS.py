"""UI package entrypoint.

This remains a compatibility shim and currently launches the Telegram UI.
"""

from .telegram.__main__ import main
if __name__ == "__main__":
    main()
