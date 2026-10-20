"""`import planetmodel` pulls in numpy and scipy only: no extra, and not
ducc0, a dependency imported where a transform is made."""
import subprocess
import sys


def test_import_is_light(extra_modules):
    code = (
        "import sys, planetmodel\n"
        f"heavy = [m for m in {extra_modules + ('ducc0',)!r} if m in sys.modules]\n"
        "assert not heavy, heavy\n"
    )
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
