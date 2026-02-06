# Active Tasks & Issues

## Priority 1: Consistency & Caching Infrastructure

### ✅ 1. Manifest Consistency Between Extraction and Matching (COMPLETED)

**Status:** Done - All environment info unified across extraction, matching, and estimation stages.

**Completed Steps:**

1. ✅ Enhanced `get_environment_info()` to capture CPU model/cores and all GPU devices
2. ✅ Added optional package versions (opencv, poselib, pycolmap, imm) to environment tracking
3. ✅ Added `output_size_mb` to estimation manifest
4. ✅ Refactored manifest creation functions to use unified environment info
5. ✅ Added `stage` identifier and optional `parent_manifest` reference for workflow chaining

**Benefits Delivered:** Complete audit trail, reproducible workflows, easier metrics aggregation

---

### ✅ 2. Homography Warp Preview (COMPLETED)

**Status:** Done - Warp visualization added to two-view homography estimation.

**Completed Steps:**

1. ✅ Created `HomographyVisualizer` class inheriting from `Viz2D` base class
2. ✅ Implemented `draw_homography_warp()` method to warp source to target using homography
3. ✅ Blend warped and target with configurable `--warp-alpha` for overlap visualization
4. ✅ Added `--show` flag to `imm-estimate homography` command for display control
5. ✅ Export warped preview as `homography_warp.png` alongside homography matrix

**Benefits Delivered:** Visual verification of registration quality, easier debugging, 3-panel visualization (source, warped, blended)

---
