# Contributing to AMDiffusionBenchmark

Thanks for your interest in contributing.

## Reporting Issues

Use GitHub Issues to report bugs or request features.
Include a clear description, reproduction steps, and your environment (OS, ROCm/CUDA version, Python version).

## Pull Request Workflow

1. Fork the repository and create a branch from `main`:

   ```bash
   git checkout -b feature/short-description
   ```

2. Make your change. Add tests and update docs if behavior changes. The changes must comply with pre-commit hooks described below.
3. Open a PR against `main`. Describe *what* changed and *why*; link any related issue.
4. Ensure CI passes and request review from the relevant [CODEOWNERS](CODEOWNERS).
By opening a PR, you agree your contribution is licensed under the terms in [LICENSE](LICENSE).

## External Contributors

This repo is part of the AMD's ROCm org. Non-AMD contributors need admin approval before being added as collaborators.

For security issues, do **not** open a public issue — see [SECURITY.md](SECURITY.md).

## Pre-commit hooks

This project uses pre-commit hooks to enforce code quality standards and maintain consistency.
These hooks automatically run before each commit to check and fix issues related to formatting, linting, and other quality checks.

### Setup

Install and configure pre-commit:

```bash
# Install pre-commit
pip install pre-commit

# Set up the git hooks
pre-commit install
```

### Usage

The hooks will run automatically on each commit.
You can also run them manually:

```bash
# Run hooks on all files
pre-commit run --all-files

# Update hooks to latest versions (recommended periodically)
pre-commit autoupdate

# Run advanced linting and security checks (optional)
pre-commit run --hook-stage manual --all-files
```

Our pre-commit configuration includes:

- Code formatting and style enforcement
- File cleanliness checks (whitespace, line endings, file validation)
- Static code analysis and linting
- Security vulnerability scanning (manual hook)
- Automatic test execution

> **Note:** Setting up pre-commit locally helps prevent CI pipeline failures, as these same checks will run when you create a pull request to the main branch.

If pre-commit prevents your commit due to failures:

1. Review the error messages
2. Fix the identified issues (many hooks will automatically fix problems)
3. Add the fixed files and try committing again

## Testing

Before opening a pull request, ensure that all new code is covered by tests.
This is essential for maintaining the project's quality and reliability.
You can run the test suite manually using:

```bash
make run_tests
```

Existing tests are run automatically as part of the pre-commit hooks when you commit changes.

### GPU-Dependent tests

The project includes integration tests that require a GPU to run.
These conditional tests verify that the pipelines work correctly on GPU hardware.
When you run `make run_tests`, these GPU-dependent tests:

- Run automatically on GPU-enabled machines
- May take several minutes to complete
- Are essential for verifying model functionality

It's highly recommended to run these tests on a GPU-enabled machine before opening a pull request.

If necessary, you can manually skip GPU-dependent tests with:

```bash
export SKIP_GPU_TESTS=1
```
