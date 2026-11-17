"""Run every check script and summarize; exit with status 1 on any failure.

Run:  python3 code/run_all_checks.py
Each code/chNN_checks.py prints its checks with OK/FAIL markers. From
Chapter 6 on, and in ch06_mutations.py, regressions.py and the solution
scripts, they end with a line 'k/n checks passed' (or 'k/n mutation
tests'); the scripts for Chapters 1-5 print the checked quantities and
mark failures with FAIL. A script fails if it exits with an error, if
its summary line reports k < n, or if it prints a FAIL marker. The
solution scripts in solutions/ are run when present (they are
distributed with the solutions manual, not in the public repository).
"""
import re
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
scripts = (sorted(HERE.glob("ch[0-9][0-9]_checks.py")) + [HERE / "ch06_mutations.py",
           HERE / "regressions.py"] + sorted(HERE.glob("solutions/sm[0-9][0-9]_checks.py")))
total_ok = total = 0
failed = []
for f in scripts:
    t0 = time.time()
    r = subprocess.run([sys.executable, str(f)], capture_output=True, text=True)
    el = time.time() - t0
    m = re.search(r"(\d+)/(\d+) (?:checks passed|mutation tests)", r.stdout)
    fails = [l for l in r.stdout.splitlines() if l.startswith("FAIL")]
    if r.returncode:
        status = "error"
    elif m:
        ok, n = int(m.group(1)), int(m.group(2))
        total_ok, total = total_ok + ok, total + n
        status = f"{ok}/{n}"
    else:
        status = "FAIL" if fails else "ran"
    bad = r.returncode != 0 or bool(fails) or (m and int(m.group(1)) < int(m.group(2)))
    if bad:
        failed.append(f.name)
    print(f"{f.name:18s} {status:>8s}  {el:7.1f} s" + (f"  {fails}" if fails else ""))
    if r.returncode:
        print("    " + "\n    ".join(r.stderr.strip().splitlines()[-5:]))
print(f"{'summarized checks':18s} {total_ok}/{total}")
if failed:
    print("FAILED:", ", ".join(failed))
    sys.exit(1)
print("all scripts passed")
