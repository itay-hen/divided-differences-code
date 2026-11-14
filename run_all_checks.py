"""Run every chapter's check script and summarize.

Run:  python3 code/run_all_checks.py
Each code/chNN_checks.py prints its checks; from Chapter 9 on they end
with a line 'k/n checks passed', while the scripts for Chapters 1-8
print the checked quantities with OK/FAIL markers.  This script reports
the summary line, or else the number of FAIL markers, and the run time.
"""
import re
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
total_ok = total = 0
for f in (sorted(HERE.glob("ch[0-9][0-9]_checks.py")) + [HERE / "ch06_mutations.py"]
          + sorted(HERE.glob("solutions/sm[0-9][0-9]_checks.py"))):
    t0 = time.time()
    r = subprocess.run([sys.executable, str(f)], capture_output=True, text=True)
    el = time.time() - t0
    m = re.search(r"(\d+)/(\d+) (?:checks passed|mutation tests)", r.stdout)
    ok, n = (int(m.group(1)), int(m.group(2))) if m else (0, 0)
    fails = [l for l in r.stdout.splitlines() if l.startswith("FAIL")]
    if r.returncode:
        status = "error"
    elif m:
        status = f"{ok}/{n}"
    else:
        status = "ran" if not any("FAIL" in l for l in r.stdout.splitlines()) else "FAIL"
    print(f"{f.name:18s} {status:>8s}  {el:7.1f} s" + (f"  {fails}" if fails else ""))
    total_ok += ok
    total += n
print(f"{'all':18s} {total_ok}/{total}")
