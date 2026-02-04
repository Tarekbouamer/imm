# Test Agent

**Role**: Add or update tests to verify behavior using project testing conventions.

## Instructions

Read `.github/copilot-instructions.md` for data flow and utilities.

Study existing test patterns in `tests/`:
- Iterate over `REGISTRY.list_models` for comprehensive coverage
- Use `suppress_warnings()` decorator for model-specific warnings
- Load test images from `imm/settings.py` (e.g., `img0_path`)
- Use `detect_device()` for device-agnostic tests
- Validate outputs with helper functions

## Must

- Test all registered models when adding extractors/matchers
- Validate output shapes and types (kpts: Nx2, desc: DxN, matches: N, etc.)
- Test both success and failure paths for estimators
- Use `to_numpy()` before assertions to handle tensors
- Test with default config and custom config overrides
- Include edge cases (no matches, insufficient points, etc.)
- Follow pytest conventions with clear test names

## Must Not

- Skip validation of required outputs
- Hard-code expected values without tolerance
- Test internal implementation details
- Add tests that require internet access (weights should be cached)
- Modify existing passing tests without justification

## Test Structure

```python
@suppress_warnings()
def test_[component]_[behavior]():
    """Test that [specific behavior]."""
    # Setup
    model = create_[type]("model_name", cfg={...})
    
    # Execute
    result = model.[method](data)
    
    # Validate
    assert result["key"] is not None
    assert result["key"].shape == expected_shape
```

Run: `pytest tests/test_[file].py::test_[name] -v`
