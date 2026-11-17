# Divided Differences — code and benchmarks

Code accompanying the book *Divided Differences: From Interpolation to
Quantum Simulation* (CRC Press, in preparation). Every numerical statement in
the book is produced or checked by a script in this repository, and every code
listing in the book is a copy of a function defined here (with comments
sometimes shortened). The checks test the cases they state; they do not
certify an algorithm for every input. Appendix C
of the book describes the scripts in detail.

## Requirements

Python 3.13 with NumPy, SciPy, mpmath and Matplotlib (versions used:
NumPy 2.2, SciPy 1.16, mpmath 1.3, Matplotlib 3.10):

```bash
pip install -r requirements.txt
```

The scripts run on one core and need no configuration.

## Contents

| File | Purpose |
|---|---|
| `chNN_checks.py` | Checks of the identities, bounds and tables of Chapter NN. From Chapter 6 on, each check prints `OK` or `FAIL` and the script ends with `k/n checks passed`. The listings of each chapter sit at the top of its script, between the markers `listings` and `helpers`. |
| `ch06_mutations.py` | Mutation tests: identities of Chapter 6 evaluated as stated and with a deliberate error, which must be detected. |
| `fig_chNN*.py` | The figures of Chapter NN, written as PDF and PNG to `../figures/` (the book's figure folder; create it when running from a standalone clone). |
| `sync_listings_chNN.py` | Copies the listings into the chapter sources (needs the book's LaTeX sources, which are not part of this repository). |
| `appC_benchmarks.py` | Writes the benchmark node sets B1–B8 to `benchmarks/` and computes their reference values. |
| `benchmarks/` | The benchmark node sets of Appendix C, one node per line, and their 60-digit reference values. |
| `regressions.py` | Cases from an external review of the draft: counterexamples to earlier versions of listings and statements, now testing the corrected behaviour. |
| `run_all_checks.py` | Runs every check script, prints a summary, and exits with status 1 if any check fails or any script errors. |

## Running the checks

```bash
python3 run_all_checks.py
```

The full run takes about 25 minutes on a laptop; the slowest scripts are those
of Chapter 6 (40-digit arithmetic) and Chapters 5, 12 and 14. Individual
chapters can be run on their own, e.g. `python3 ch09_checks.py`.

The scripts that check the instructor's solutions manual are not included.

## Citation

See `CITATION.cff`. The release tagged for the printed edition of the book
will be marked as such.

## License

MIT; see `LICENSE`.
