# CLI Inconsistencies - Outstanding Tasks

## Argument Naming (Remaining)

- `--det_thd` in `_extract.py` vs `--reproj_thd` in `_estimate.py` (homography) vs `--threshold` in `_estimate.py` (relative_pose) vs `--match_thd` in `_match.py`

**Tasks:**

- [ ] Standardize threshold naming across all scripts
- [ ] Rename `--det_thd` → `--det_threshold` in `_extract.py`
- [ ] Rename `--match_thd` → `--match_threshold` in `_match.py`

## Option Order

**Tasks:**

- [x] Establish standard option order: input paths → model options (matcher/extractor) → processing options (thresholds, sizes) → output options → device/performance flags → help ✓
- [x] Reorder options in `_extract.py` to match standard ✓
- [x] Reorder options in `_match.py` to match standard ✓
- [x] Reorder options in `_estimate.py` to match standard ✓

## Backend/Solver Options

**Tasks:**

- [ ] Document available solvers for each estimation type in help text

---

## Completed Tasks

### Argument Naming (Completed)

- ✓ Standardized image size param to `--resize` (matches `load_image()` function parameter)
- ✓ Renamed `--max_img_size` → `--resize` in `_extract.py` and `_match.py`
- ✓ Renamed `--max_size` → `--resize` in `_estimate.py`

### Flag Naming (Completed)

- ✓ `--force_cpu` consistent across `_extract.py`, `_match.py`, `_estimate.py`
- ✓ `--visualize` / `--viz` consistent across all scripts
- ✓ Added `--viz` as short alias for `--visualize`

### Argument Types (Completed)

- ✓ Kept positional arguments for pair-based commands (match, estimate)
- ✓ Kept option for single/batch input in extract
- ✓ Design intentionally different based on command purpose

### Default Values (Completed)

- ✓ Standardized `--resize` default to `640` across all scripts
- ✓ Standardized output parameter to `--output` across all scripts
- ✓ `--num_workers` at `4` (only in _extract.py where needed)

### Missing Options (Completed)

- ✓ Renamed `--save_path` → `--output` in `_match.py`
- ✓ Renamed `--output_dir` → `--output` in `_extract.py` and `_estimate.py`
- ✓ Added `--visualize` to `_extract.py`
- ✓ Added `--max_keypoints` to both `_estimate.py` commands (homography and relative_pose)
  - Default: -1 (keeps all keypoints)
  - Passed to Matching class for consistency with extract and match commands

### Argument Naming (Completed)

- ✓ Renamed homography reprojection threshold option to `--reproj_thd`
  - Kept `--threshold` for `relative_pose`

### Backend/Solver Options (Completed)

- ✓ Kept backend/solver options specific to estimation commands

---

## Summary of Priority Tasks

### High Priority (Breaking Changes - Needs Coordination)

1. Standardize threshold parameter names (`--det_threshold`, `--match_threshold`)

### Low Priority (Nice to Have)

1. Add usage examples to help texts
