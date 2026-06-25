"""Write a VNN-LIB version note next to every results.csv.

A tool reports results for exactly one VNN-LIB version per benchmark, but the
version is only implicit in the network/property paths of ``results.csv`` (there
is no version column). This script makes the provenance explicit: for every
``results.csv`` it derives, per benchmark, which VNN-LIB version that tool ran
(from the version directory in the path -- the same logic scoring uses) and writes
a ``vnnlib_version.txt`` note in the same folder.

It handles both result layouts produced/consumed by the pipeline:

    <tool>/<year>_<benchmark>/results.csv     # per benchmark (orchestrator export)
    <tool>/results.csv                        # flat per-tool merge (aggregate scoring)

In the per-benchmark layout each note is a single line (that benchmark's version);
in the flat layout the note lists one line per benchmark.

Run it from the SCORING directory (or anywhere) after the result folders exist:

    python write_version_notes.py                 # scans ../**/results.csv
    python write_version_notes.py --root /path/to/vnncomp2026_results
    python write_version_notes.py --dry-run       # print, do not write
"""

import argparse
import csv
import glob
import os
from collections import defaultdict

from benchmark_instances import benchmark_version_from_network_field

# Column layout of results.csv (mirrors ToolResult in process_results.py).
CATEGORY = 0
NETWORK = 1

# results.csv stores the benchmark name without the competition year; scoring
# prepends it before resolving the version from the path, so we do the same.
YEAR = "2026"


def versions_per_benchmark(results_csv):
    """Map benchmark name -> sorted list of VNN-LIB versions seen in the rows."""
    seen = defaultdict(set)
    with open(results_csv, newline="") as handle:
        for row in csv.reader(handle):
            if not row:
                continue
            category = row[CATEGORY]
            network = row[NETWORK]
            version = benchmark_version_from_network_field(f"{YEAR}_{category}", network)
            seen[category].add(version or "unknown")
    return {cat: sorted(versions) for cat, versions in sorted(seen.items())}


def render_note(label, per_benchmark):
    lines = [
        f"# VNN-LIB version for {label}",
        "# Auto-generated from results.csv by SCORING/write_version_notes.py.",
        "# One line per benchmark: '<benchmark>  <vnnlib-version>'.",
        "",
    ]
    for benchmark, versions in per_benchmark.items():
        if len(versions) == 1:
            lines.append(f"{benchmark}  {versions[0]}")
        else:
            # A single tool should run one version per benchmark; flag if not.
            lines.append(f"{benchmark}  MIXED:{','.join(versions)}  # unexpected, please check")
    return "\n".join(lines) + "\n"


def find_results_csvs(root):
    matches = glob.glob(os.path.join(root, "**", "results.csv"), recursive=True)
    # Never treat anything under a SCORING/ directory (logs, fixtures) as a tool result.
    return sorted(
        path for path in matches
        if "SCORING" not in os.path.relpath(path, root).split(os.sep)
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    here = os.path.dirname(os.path.realpath(__file__))
    parser.add_argument(
        "--root",
        default=os.path.dirname(here),
        help="results-repo root to scan for results.csv (default: parent of SCORING)",
    )
    parser.add_argument("--dry-run", action="store_true", help="print notes, do not write")
    args = parser.parse_args()

    results_files = find_results_csvs(args.root)
    if not results_files:
        print(f"no results.csv found under {args.root!r}")
        return

    for results_csv in results_files:
        result_dir = os.path.dirname(results_csv)
        # Label by the path relative to the root, e.g. "abcrown/2026_acasxu_2023".
        label = os.path.relpath(result_dir, args.root).replace(os.sep, "/")
        per_benchmark = versions_per_benchmark(results_csv)
        note = render_note(label, per_benchmark)
        note_path = os.path.join(result_dir, "vnnlib_version.txt")

        if args.dry_run:
            print(f"==== {note_path} ====")
            print(note)
        else:
            with open(note_path, "w") as handle:
                handle.write(note)
            print(f"wrote {note_path} ({len(per_benchmark)} benchmark(s))")


if __name__ == "__main__":
    main()
