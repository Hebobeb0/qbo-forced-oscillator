# Figures communicating the additive-forcing conclusion

Use `conclusion_summary.png` as the main figure. Its four panels establish the positive timing effects and the limits of an entrainment interpretation. `05_interval_distributions.png` is a useful companion because it shows the finite widths and offsets of the actual interval modes. PNG exports have a white background; SVG exports retain editable vector elements and text.

## Suggested main-figure caption

**Additive forcing organizes cycle timing without establishing sustained two-period locking in the tested cases.** Simulations use dX = V dt and dV = -(X + epsilon V) dt + sqrt(epsilon(sigma² + 2rX²)) dB + delta sin(omega t) dt, with epsilon = 0.05, sigma = 1, integration timestep 0.0025, total duration 4000 and burn-in 1000. Cycle intervals N are expressed in forcing periods, using interpolated upward zero crossings accepted after an excursion below -0.25 RMS(X). **(A)** Crossing-phase distributions become concentrated as forcing increases. Each curve is the mean of 16 independently simulated, individually normalized histograms; shaded bands give pointwise 95% confidence intervals across seeds. **(B)** Near-integer interval probabilities increase, with the one-period band growing strongly at higher forcing. The two-period band does not increase monotonically with forcing. Bands satisfy |N - n| < 0.1; the black curve includes proximity to any integer, so it need not equal the sum of only the three displayed bands. Points and bars are seed means and 95% confidence intervals, respectively. Panels A and B use r = 0.5 and omega = 7/3. **(C)** The slope of the branch near N = 2 decreases similarly for r = 0 and r = 0.5, remaining above zero. At r = 0, direct forcing and unforced stochastic trajectories plus the deterministic response produce identical crossing counts and event times to numerical precision. Error bars are 95% bootstrap intervals obtained by resampling eight whole seed curves and re-extracting the branch. Slopes are fitted over omega = 1.9–2.6. **(D)** Changes in the unwrapped generalized phase (2 phi - omega t)/(2 pi) show systematic drift. Phi is the geometric phase atan2(-V,X). All paths are aligned to zero phase change at the start of the displayed interval; this does not assert equal absolute initial phases. Curves and shading show means and pointwise 95% confidence intervals across all eight saved independent focal runs, displaying their first 300 forcing periods after burn-in, without trajectory selection. The dashed zero-drift reference is a necessary frequency-matching condition, not by itself a sufficient test of locking.

Confidence intervals quantify variation across independent simulation seeds. They are not simultaneous confidence bands over all times, bins or parameter values. The figures use fresh follow-up simulations from the supplied SDE, rather than the catalogue's original event files. They support phase-selective crossing timing and near-integer interval enrichment, but do not uniquely establish a skipped-opportunity mechanism or rule out brief locking episodes or locking elsewhere in parameter space.

## Suggested companion-figure caption

**Forcing redistributes intervals between broad modes.** At r = 0.5 and omega = 7/3, the interval density shifts from a dominant peak near the unforced cycle length toward modes near one and two forcing periods as delta increases. Vertical guides mark integer N. The finite-width peak associated with N near 2 is offset from the exact integer, illustrating why proximity to an integer should not be interpreted as exact quantization or sustained locking. Curves average 16 seed-normalized histograms, using bins of width 0.025 in N and Gaussian smoothing with standard deviation one bin. Shading represents pointwise 95% confidence intervals across seeds. Histograms are normalized over N in [0,6]; the displayed range is [0.5,3.5].

## Data and panel files

| Panel | Individual figure stem | Plotted data |
|---|---|---|
| A | 01_crossing_phase | crossing_phase_density.csv |
| B | 02_integer_interval_probabilities | interval_band_probabilities.csv |
| C | 03_linear_control | branch_slopes.csv |
| D | 04_relative_phase_drift | relative_phase_drift.csv |
| E | 05_interval_distributions | interval_density.csv |

The phase-density estimates use 48 equal-width bins around a full forcing cycle, Gaussian smoothing with standard deviation 0.7 bins, and periodic boundary handling. Band probabilities are calculated from individual intervals, not from these smoothed density estimates. Branch-extraction definitions and simulation details are in the earlier full report, `outputs/forcing_followup/report.md`.

The included figure-building source documents every transformation. Rebuilding it requires the earlier follow-up event archive, summary tables, saved focal trajectories, Python with NumPy/SciPy/ReportLab, and the indicated Windows fonts. The SVG-to-PNG script uses Sharp. No new simulations were performed to create these figures.
