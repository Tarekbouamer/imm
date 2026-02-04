# IMM (ImMatch) — Copilot Instructions

## What this repo is
Computer vision library for image matching, feature extraction, and geometric estimation.
Unified interfaces for extractors, matchers, estimators.

## Architecture (must preserve)
- Extractors: imm/extractors/
- Matchers: imm/matchers/
- Estimators: imm/estimators/
- Registry pattern: imm/registry/ (decorator registration + factory creation)

## Critical invariants (do not break)
- Models register via registry decorators with default_cfg.
- Weight loading uses factory load_model_weights(), caching to hub/.
- Matchers must respect required_inputs:
  - Sparse: kpts/desc pairs
  - Dense: raw images
- Use @torch.inference_mode() for extract/match paths.
- Estimator outputs include success flag; callers must check it.

## CLI entry points
- imm-extract, imm-match, imm-estimate, imm-gui

## Dev commands (verify before done)
- make dev / make test / make lint / make format
