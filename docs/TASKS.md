# Flow Line Improvements

This document lists tasks for enhancing flow line visualization and the backend infrastructure. See `2102.06824v2.pdf` for the theoretical background and goals of this project.

## Backend
- [x] Review momentum flow line generation in `warp.analyzer.get_momentum_flow_lines`.
  - Incorporate guidelines from `docs/2102.06824v2.pdf` to ensure physically meaningful trajectories.
  - Consider adaptive step sizes or termination conditions based on energy thresholds.
- [x] Add unit tests for the improved flow line generator.
- [x] Document new arguments and behaviour in the module docstring.

## Notebook
- [x] Update `notebooks/intro.ipynb` to demonstrate the enhanced flow line logic.
  - Include visual comparisons of original and improved flow lines.
  - Provide references to relevant sections of the PDF.
- [x] Validate notebook cells so that example graphs show non-empty trajectories.
- [x] Document common errors (e.g. singular matrices) and provide working
  Alcubierre parameters.
- [x] Extend the notebook with a section plotting energy conditions using
  `plotly` for a valid Alcubierre warp bubble.

## Miscellaneous
- [x] Add examples to the README that showcase how to run the notebook and interpret the results.
