# Reproducing the follow-up experiments

Start with `report.md` for findings, methods and limitations. The numerical CSV files contain all reported summaries and seed-level estimates. The figures are static exports at 1660 pixels wide.

## Saved raw events

`events.npz` contains 3,072 primary simulated runs. Each row of `parameters` is `(r, delta, omega, seed)`. Events for row `j` occupy slice `offsets[j]:offsets[j+1]` of `time`, `minimum_since_previous`, and `crossing_velocity`. `statistics[j,0]` is that run's post-burn-in RMS(X). Other statistics are described in `manifest.json`.

To apply a Schmitt factor `c`, accumulate the minimum across raw crossings, accepting a crossing when that accumulated minimum is at most `-c*rms_X`, then reset the accumulated minimum to zero. The `accepted` function in `work/analyse_experiments.py` implements the rule. Raw events are already interpolated at integration resolution. Data outside the [1000,4000] analysis interval are omitted.

## Run the full experiments again

The supplied implementation targets 64-bit Windows with GCC and Python. Dependencies: NumPy, SciPy, Pillow (version numbers in the manifest). Plotting uses Windows Segoe UI fonts. `run_experiments.py` uses `C:/msys64/ucrt64/bin` as the DLL search directory; change that line if GCC is installed elsewhere. The published archive contains source, not a precompiled executable.

From the extracted archive root:

```powershell
gcc -O3 -shared -o work/forcing.dll work/forcing.c -lm
python work/run_experiments.py
python work/validate_superposition.py
python work/analyse_experiments.py
python work/opportunities.py
python work/verify_numerics.py
python work/make_figures.py
```

The run script skips simulation files that already exist. The archived `work/runs/restore_*.npz` files contain the original raw perturbation results, so those are reused unless moved aside. The branch grid and focal trajectories will be regenerated on a fresh extraction. Full focal trajectories are retained by the run script for the sampling-sensitivity analysis; they are omitted from the archive to keep it compact. Raw crossing records are included separately in `events.npz` and do not require regeneration for event-level analyses.

The C implementation uses negative `r` only as an internal flag to construct the exact linear superposition during validation; it is never a physical negative-noise parameter. Normal simulation parameters are r = 0 and r = 0.5. No amplitude rescaling occurs.

The code, simulation seeds, parameters and source hashes are included for auditing. This is an independent follow-up to the catalogue, not the catalogue's original simulation implementation.
