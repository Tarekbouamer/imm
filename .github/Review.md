# Review Tasks

## Overview

Active review items and validation checklists consolidated from Task.md.

## Priority 2: Metrics Validation & Testing

### Verify Metrics Against Common Practices

**Problem:** Metrics implementations (pose error, epipolar error, AUC) need validation against established benchmarks and common practices to ensure correctness.

**Steps:**

1. Verify pose error computation against standard formulations (rotation/translation angular errors)
2. Validate epipolar distance methods (Sampson vs symmetric) against reference implementations
3. Test AUC computation against known datasets with ground truth
4. Run metrics on ScanNet dataset to verify results match expected benchmarks
5. Compare with published baselines from image matching literature
6. Add unit tests with known ground truth values

**Benefits:** Confidence in metric correctness, reproducible benchmarks, compatibility with standard evaluation protocols

---
