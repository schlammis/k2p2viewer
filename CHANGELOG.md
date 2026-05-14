# Changelog

All notable changes to k2p2viewer are documented here.

---

## [1.2.6] – 2026-05-14

### Added

- **Allan deviation tab** — log-log plot of Allan deviation vs. averaging number *m*, computed with the overlapping estimator from `mystat.AllanDeviation`. Each point is shown with its own uncertainty bar. The tab sits between the Report and Diagnostics tabs.
- **ppm / µg toggle** — checkbox on the Mass tab switches the right-hand axis between µg and ppm relative to the nominal mass.
- **g / mg toggle** — checkbox on the Mass tab switches the left-hand axis, reference mass field, and displayed mass values between mg and g.
- **Legend on mass plot** — the blue band (overall uncertainty, k-coverage) and red band (Type A uncertainty) are now labelled in a legend on the mass graph.
- **Difference from nominal block** — below the uncertainty stats on the Mass tab, the deviation from the nominal reference mass is shown in µg together with the combined uncertainty and coverage factor.
- **Copy table to clipboard** — right-clicking on the left-hand data table or its tab header opens a context menu with "Copy table to clipboard". The data is placed on the clipboard as tab-separated values (TSV) ready for pasting into Excel.
- **Force scan all directories** — a checkbox in the toolbar instructs the scanner to ignore any existing `k2readerror.dat` sentinel files and attempt to re-read every run directory. The setting is persisted in `config.ini`.
- **Auto-populate reference mass** — when opening a run the reference mass field is populated automatically from the `NominalMassGrams` key in the measurement `config.ini` (converted to mg). When that key is absent, the nearest standard mass (1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10 000, 20 000, 50 000 mg) is selected using log-scale distance. The chosen value is persisted in the application `config.ini`.

### Fixed

- **Reference mass line intermittently missing** — the dotted reference line on the mass graph was not drawn when switching between runs because the stored value was cleared on each reload. The plot now reads the spinbox value directly so the line always appears when a reference mass is set.
- **Non-standard VerticalityDate strings** — date strings such as `ABC DE FGHI` in a measurement `config.ini` caused an unhandled exception and wrote a `k2readerror.dat` for the whole run directory. The parser now falls back to a placeholder date (0) and logs a warning instead of aborting.
- **Stale k2readerror.dat auto-deleted** — if a run directory that previously failed to parse now reads successfully (e.g. after using *force scan*), the old sentinel file is removed automatically.
- **k2readerror.dat now includes directory and traceback** — the error file written when a run cannot be parsed now contains the full directory path and a Python traceback to make diagnosis easier.
- **PRTData.dat fault handling** — the environmental data loader now distinguishes three failure modes and reports each clearly below the mass statistics:
  - *File missing* — uses standard values (T = 20 °C, P = 999.9 hPa, rH = 40 %) and shows an amber warning.
  - *All-zero rows* — same fallback; warning explains the cause.
  - *Read error* — same fallback; the exception message is included in the warning.
  - Single-row dropouts (one timestamp with all zeros) are silently skipped; the number of skipped rows is shown as a minor note rather than a full warning.
- **Drop-first-N fix** — dropping 0, 1, or 2 initial measurements now behaves correctly in all cases.
- **Outlier exclusion error handling** — if the Huber robust estimator fails (e.g. fewer than 7 points), the failure is reported to the runtime log and all points are kept, rather than crashing silently.

---

## [1.2.3] – 2026-05-08

- SQLite result cache to avoid re-computing unchanged runs on startup.
- Sinc-correction option for the velocity fit.
- Configurable polynomial fit order.

---

## [1.2.x] – earlier

- Tabbed UI with Mass, Report, Uncertainty, Environmental, and Diagnostics panels.
- Outlier exclusion using Huber M-estimator (5 σ, requires N > 6).
- Excel export of mass results.
- Multiple balance configurations stored in `config.ini`.
- Splash screen and PyInstaller single-folder distribution.
