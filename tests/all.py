# pylint: disable=no-member, arguments-differ, missing-docstring, invalid-name
import glob
import os
import subprocess
import sys

d = os.path.dirname(os.path.abspath(__file__))
python = sys.executable if sys.executable else "python"
print('interpreter is {}'.format(python))

for f in sorted(glob.glob(os.path.join(d, "*_tests.py"))):
    print('execute {}'.format(os.path.basename(f)))
    out = subprocess.run([python, f])
    if out.returncode != 0:
        break
