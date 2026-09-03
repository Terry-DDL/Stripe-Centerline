# Stripe Centerline

A desktop computer-vision tool for detecting and measuring adjacent stripe centerlines from grayscale imagery.

**Final internship handoff: September 2026**

This repository contains the production Tk desktop application and the frozen Stage 3.1 / cross-image correction v1.1 detection pipeline. The detector uses conservative verification: it reports only centerlines supported by the available image evidence and safety checks.

- **Final source commit:** `45c1bd34bd56fb18e87d072324be376eba3815d1`
- **Recommended environment:** 64-bit Python 3.11
- **Desktop entry point:** `tools/desktop_app.py`

## Windows setup

Open Command Prompt in the project directory and run:

```bat
py -3.11 -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python tools\desktop_app.py
```

For later launches, only the final command is required. The application can use the sample images in `images/` or open another image through the desktop interface.

## macOS setup

Open Terminal in the project directory and run:

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python tools/desktop_app.py
```

## Key files

- `tools/desktop_app.py` — production Tk desktop application, image selection, point selection, and result display.
- `tools/stage3_desktop_runtime.py` — desktop result adaptation, formal result construction, persistence, and overlay output.
- `tools/cross_image_correction_prototype_v1_1.py` — frozen cross-image correction v1.1 detection pipeline used by the production desktop application.
- `tools/separator_path_prototype.py` — separator candidate generation and path tracking.
- `tools/basin_graph_joint_prototype.py` — basin graph construction and joint safety verification.
- `tools/raw_local_pitch_prototype_v3.py` — local theoretical pitch estimation.
- `src/config.py` and `configs/` — application and pitch-estimation configuration.
- `tests/` — desktop, Stage 3.1, detector, safety, and regression tests with their required test data.

## Current behavior

- When both adjacent centerlines are safely verified, the application reports bilateral output and centerline spacing.
- When only one side can be safely verified, the application preserves that side as an independent **only-left** or **only-right** result; the other side is shown as unavailable.
- When the available evidence is insufficient for either side, the result is shown as unavailable. This is conservative verification, not an application failure.
- Centerline spacing is reported only when both sides are available. It remains unavailable for one-sided results.

## Testing

After installing the dependencies, run the relevant regression suite from the project directory:

```bat
.venv\Scripts\python -m unittest tests.test_separator_path_prototype tests.test_basin_graph_joint_prototype tests.test_cross_image_correction_prototype_v1_1 tests.test_stage3_desktop_runtime tests.test_desktop_app tests.test_adjacency_safety_regression tests.test_windows_benchmark_runner tests.test_compare_windows_benchmarks
```

On macOS, replace `.venv\Scripts\python` with `.venv/bin/python`.

Two development provenance tests read the current Git commit and therefore require `.git` metadata. The source ZIP intentionally does not include repository history, so those two tests may report errors when the full command is run directly from an extracted ZIP. This is unrelated to detector correctness. The final clean Git checkout completed **120/120 regression tests successfully**.

## Source package contents

The source handoff ZIP intentionally excludes:

- `.venv`
- `__pycache__`
- build and `dist` artifacts
- benchmark result packages
- debug outputs
- historical runtime outputs

`benchmark_assets/` contains only the fixed input image required by the Stage 3.1 edge regression tests; it does not contain benchmark results.
