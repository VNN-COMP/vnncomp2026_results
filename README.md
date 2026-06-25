# vnncomp2026_results

Measurements and scoring for VNN-COMP 2026.

## Layout

Results are stored **per tool, per benchmark**. The orchestrator's export writes one
folder per `(tool, benchmark)` pair:

```
vnncomp2026_results/
├── SCORING/                          # scoring + counterexample validation (run from here)
├── <tool>/
│   └── <year>_<benchmark>/           # e.g. abcrown/2026_acasxu_2023/
│       ├── results.csv               #   this tool's measurements for this benchmark
│       ├── *.counterexample.gz       #   witnesses the tool produced (sat instances)
│       ├── *.counterexample.check.json   # validator verdicts (added by scoring)
│       └── vnnlib_version.txt        #   which VNN-LIB version this run used
└── ...
```

`SCORING/process_results.py` can process a single benchmark folder with
`--single-benchmark <tool>/<year>_<benchmark>/results.csv` (this is what the orchestrator
runs per benchmark), a whole tool with `--single-tool`, or the full field via the
aggregate glob (`Settings.CSV_GLOB`, which expects a flat per-tool `results.csv`).

## `results.csv` format

One row per benchmark instance, no header:

| col | field          | notes                                                        |
|-----|----------------|--------------------------------------------------------------|
| 0   | `category`     | benchmark name **without** version (e.g. `acasxu_2023`)      |
| 1   | `network`      | path to the ONNX model — **contains the version directory**  |
| 2   | `property`     | path to the `.vnnlib` query                                  |
| 3   | `prepare_time` | seconds                                                      |
| 4   | `result`       | `holds` / `violated` / `timeout` / `error` / `unknown` / ... |
| 5   | `run_time`     | seconds                                                      |

## VNN-LIB version

Each `(tool, benchmark)` run uses exactly **one** VNN-LIB version. The version is **not**
a separate column — it is implicit in the `network` / `property` paths, which point into
the benchmark repo's `benchmarks/<category>/<version>/...` directory (e.g.
`.../acasxu_2023/2.0/onnx/...`).

A `vnnlib_version.txt` note is generated next to each `results.csv` by

```
python SCORING/write_version_notes.py        # scans ../**/results.csv
```

so the version a tool used for a benchmark is visible at a glance without parsing the CSV.
(A tool should run one version per benchmark; a benchmark seen under both versions is
flagged `MIXED:` for review.)

Scoring is **cross-version**: `process_results.py` compares tools purely by
`(category, instance index)`, so a tool that ran the 1.0 instances and a tool that ran
the 2.0 instances are scored head-to-head on the same logical instance. This relies on
the 1.0 and 2.0 `instances.csv` files listing the same instances in the same order;
`compare_results` asserts that invariant per category. Counterexamples are validated
against the version each row actually used (1.0 vs 2.0 validators are selected from the
path).
