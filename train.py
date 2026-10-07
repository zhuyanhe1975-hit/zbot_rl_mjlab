"""Register local Zbot tasks and invoke the MJLab training CLI."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

if __name__ == "__main__":
    from mjlab.scripts.train import main

    import zbot_rl_mjlab  # noqa: F401

    main()
