# Copilot Instructions

## Project Overview
Computer vision library for image matching, feature extraction, and geometric estimation with unified component interfaces.

## Architecture Patterns (preserve)

**Component Organization:**
- Feature extractors and their implementations
- Matchers and matching strategies
- Estimators for geometric computation
- Base classes with consistent interfaces
- CLI tools with consistent patterns
- Public API with factory functions
- Registry system for dynamic component registration

**Key Principles:**
- Models register via decorators with configuration
- Weight management includes caching and auto-download
- Components validate required inputs before execution
- Inference uses inference mode decorators
- Operations return validated outputs with success indicators

**Critical Contracts:**
- Registry pattern: decorator registration with config dictionaries
- Matchers respect input type contracts (sparse vs dense)
- Estimators include success flags in outputs
- Configuration merges: defaults → user → kwargs
- Outputs converted to standard types (numpy arrays)

## Common Patterns

- Use `@REGISTRY.register()` for model registration
- Include `default_cfg` with configuration and weight URLs
- Validate inputs before processing
- Use inference mode decorators for evaluation
- Return numpy arrays from public methods
- Add docstrings with Args/Returns/Raises sections
- Use device-agnostic utilities for hardware abstraction

## Code Style

- **No module-level docstrings** - Do not add triple-quoted strings at the beginning of files
- Keep imports at the top without any docstring before them
- Use inline comments for file-level context if needed

## Verification Commands
Check `Makefile` or `pyproject.toml` for available dev commands (typically: test, lint, format)
