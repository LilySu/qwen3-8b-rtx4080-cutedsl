# bench/results/

Append-only JSON files produced by `bench/runner.py` and `bench/measure_peaks.py`.

## Rules

1. **Never edit these files by hand.** They are the raw measurement record.
2. **Commit every file.** JSON is small; the full history is the point.
3. **Update ITERATIONS.md** after each run — copy the relevant `time_us` into the table.

## File types

`peaks_<ts>.json` — empirical SM89 peaks (bandwidth, BF16 TFLOPS, INT8 TOPS).  
Regenerate this after any hardware or driver change, or after pinning a new TGP.

`run_<ts>.json` — output of one `bench/runner.py` invocation.  
Contains one record per (kernel, version, regime) combination that was run.

## Reading a run file

Each record has the fields from `kernels/_base.Metrics.as_dict()`:

```json
{
  "kernel": "rope",
  "version": "v0_pytorch",
  "regime": "decode",
  "shape": [1, 1, 32, 128],
  "dtype": "bfloat16",
  "time_us": 45.2,
  "bandwidth_gbs": 12.3,
  "tflops": null,
  "pct_peak_bw": 3.1,
  "pct_peak_flops": null,
  "timestamp": "2026-07-09T14:22:00"
}
```

`pct_peak_bw` and `pct_peak_flops` are relative to the constants in `kernels/_base.py`.
Update those constants from `peaks_*.json` before comparing versions.
