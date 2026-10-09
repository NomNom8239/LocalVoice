"""Single LocalVoice entry point for new pipeline capabilities."""
from __future__ import annotations

import sys


def main() -> int:
    if len(sys.argv) >= 2 and sys.argv[1] == "asr-pilot":
        from .transcription.pilot import main as pilot_main
        return pilot_main(sys.argv[2:])
    if len(sys.argv) >= 2 and sys.argv[1] == "style-pilot":
        from .style.pilot import main as pilot_main
        return pilot_main(sys.argv[2:])
    print("Usage: python -m localvoice {asr-pilot|style-pilot} ...", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
