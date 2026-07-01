# AGENTS.md

This is a Python + OpenCV image-processing project.

## Core rules
- Work in small steps.
- Do not make large unexplained changes.
- Before implementation, explain the plan first.
- After implementation, explain changed files, functions, parameters, and known failure cases.
- Do not rewrite unrelated files.

## Project goal
Detect the centerline of the black region near a configurable vertical reference line in grayscale stripe images.

## Assumptions
- The default reference line is the image vertical center: x = width * 0.5.
- The reference line must remain configurable for future UI adjustment.
- The default ROI is centered around the image center.
- ROI width and height must remain configurable.
- White stripes may be broken, noisy, or uneven. Do not assume they are continuous.

## Coding rules
- Keep code simple and readable for an OpenCV beginner.
- Put tunable parameters in a config object or config.py.
- Save debug images for every major processing step.
- Do not add UI, automatic ROI detection, deep learning, or complex algorithms unless explicitly requested.