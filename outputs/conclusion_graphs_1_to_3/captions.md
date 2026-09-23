# Additive forcing: amplitudes 1–3

The revised figures show delta = 1, 1.5, 2, 2.5 and 3, alongside the unforced delta = 0 reference. Delta = 4 is omitted. Where amplitude is the horizontal axis, the five amplitudes appear as measurement points rather than separate curves.

The model is dX = V dt, dV = -(X + epsilon V) dt + sqrt(epsilon (sigma^2 + 2 r X^2)) dB + delta sin(omega t) dt, interpreted in the Ito sense. Epsilon = 0.05 and sigma = 1. The focal comparisons use r = 0.5 and omega = 7/3. Intervals are expressed as N = interval / forcing period, with forcing period 2 pi / omega.

## Figure captions

**01: Crossing phase.** Increasing forcing concentrates accepted upward crossings at preferred phases of the drive. Curves show smoothed phase densities, averaged over 16 independent seeds per amplitude; shading gives pointwise 95% confidence intervals across seeds.

**02: Integer interval probabilities.** The fraction of intervals within 0.1 forcing periods of an integer increases across the sampled amplitudes. Separate curves show bands around N = 1, 2 and 3. Error bars give 95% confidence intervals across 16 seeds. Integer-adjacent intervals alone do not demonstrate sustained subharmonic locking.

**03: Linear control.** The slope of the tracked interval-density branch decreases with forcing amplitude in both the r = 0 linear stochastic model and the r = 0.5 model. Slopes are fitted across omega = 1.9–2.6; confidence intervals use 300 bootstrap resamples of the eight complete seed curves. Thus branch flattening is also present in a model admitting exact deterministic-response plus unforced-noise superposition.

**04: Relative phase.** Changes in the geometric 2:1 relative phase, (2 phi - omega t)/(2 pi), over the first 300 drive periods after burn-in. Curves average eight independent seeds; shading is a pointwise 95% confidence interval. The delta = 1.5 mean is close to zero net drift, but a nearly flat ensemble mean alone is insufficient evidence of phase locking: individual phase fluctuations and slips still need examination.

**05: Interval distributions.** Smoothed interval densities for all five forcing amplitudes and the unforced reference. Vertical lines mark integer forcing periods. Shading gives pointwise 95% confidence intervals across 16 seeds. These show changes in timing distributions, without identifying persistent integer-period trajectories by themselves.

The four-panel summary combines figures 01–04. Figure 05 is a companion showing the complete distribution shapes over the displayed interval range.

## Computation and files

Crossings use a hysteresis acceptance threshold of 0.25 times the trajectory RMS of X. Simulations use time step 0.0025, total duration 4000 and burn-in 1000. Delta = 2.5 was freshly simulated, using the same design and seed sets as the earlier amplitudes: 496 frequency-grid runs and 16 additional focal runs. Linear-control superposition was also checked for the new amplitude; crossing counts agreed and the maximum event-time discrepancy was approximately 5.46e-12. Additional provenance is recorded in new_simulations.json and delta_2p5_superposition_checks.csv.

Each figure is supplied as PNG and editable vector SVG. CSV files contain the plotted estimates and confidence limits or the supporting summaries. The ZIP contains these figures, data and captions. It is a figure-and-summary-data package, not an archive of every simulated trajectory.
