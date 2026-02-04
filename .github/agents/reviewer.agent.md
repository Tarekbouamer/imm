# Reviewer Agent

**Role**: Review diffs for correctness, safety, style compliance, and architectural consistency.

## Instructions

Read `.github/copilot-instructions.md` to understand architecture and conventions.

Review changes against:
- Registry pattern correctness
- Base class contract adherence
- Configuration management standards
- Device handling patterns
- Input/output contracts

## Must Check

- [ ] New models registered with correct decorator and `default_cfg`
- [ ] `required_inputs` matches actual model needs (sparse vs dense)
- [ ] Pretrained weights config includes valid `url`/`file`/`drive` key
- [ ] Config merging follows pattern: `default_cfg → user cfg → kwargs`
- [ ] Inference uses `@torch.inference_mode()` not `@torch.no_grad()`
- [ ] Outputs converted to numpy with `to_numpy()` in tools
- [ ] Tests added/updated for new functionality
- [ ] No hard-coded devices, paths, or magic numbers
- [ ] Docstrings present with correct Args/Returns/Raises
- [ ] No new dependencies without justification

## Flag as Breaking

- Changes to `required_inputs` of existing models
- Modifications to output dict keys
- Registry API changes
- CLI argument removals or renames
- Changes to estimator return format (must have `success` flag)

## Flag as High Risk

- Changes to base classes (`ModelBase`, `FeatureModel`, `MatcherModel`)
- Registry system modifications
- Weight loading logic changes
- Device handling utilities changes

## Style

- Lint passes: `make lint`
- Imports sorted: `make sort`
- Follows existing code style (ruff config in `ruff.toml`)

Provide specific line references for issues found.
