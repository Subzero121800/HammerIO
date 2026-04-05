# Contributing to HammerIO

Thank you for your interest in contributing to HammerIO! This project is maintained by
ResilientMind AI (Joseph C McGinty Jr).

## Getting Started

1. Fork the repository
2. Clone your fork:
   ```bash
   git clone https://github.com/YOUR_USERNAME/hammerio.git
   cd hammerio
   ```
3. Install in development mode:
   ```bash
   pip install -e ".[dev]"
   pre-commit install
   ```
4. Create a feature branch:
   ```bash
   git checkout -b feature/your-feature-name
   ```

## Development Environment

### Requirements
- Python 3.10+
- NVIDIA GPU with CUDA support (optional for CPU-only development)
- FFmpeg installed

### On Jetson
```bash
# JetPack 6.x (L4T R36)
sudo apt-get install python3-pip ffmpeg
pip install -e ".[all]"
```

### On Desktop
```bash
# CUDA toolkit must be installed
pip install -e ".[all]"
```

## Code Standards

- **Type hints**: All public functions must have type annotations
- **Docstrings**: Google-style docstrings on all public methods
- **Formatting**: Black (line-length=100)
- **Linting**: Ruff
- **Type checking**: mypy (strict mode)
- **Tests**: pytest with 80%+ coverage target

Run checks:
```bash
black --check hammerio/
ruff check hammerio/
mypy hammerio/
pytest --cov=hammerio
```

## Pull Request Process

1. Ensure all tests pass
2. Update documentation if needed
3. Add a CHANGELOG entry
4. Submit PR with clear description
5. Wait for review

## Reporting Issues

Use GitHub Issues with the provided templates:
- **Bug reports**: Include hardware info (`hammer info --hardware`), steps to reproduce, and error output
- **Feature requests**: Describe the use case and expected behavior

## Architecture Notes

- All GPU operations MUST have CPU fallback
- Never load entire files into RAM — stream in chunks
- Every fallback must be logged with reason and severity
- Test on both Jetson and desktop CUDA when possible

## License

By contributing, you agree that your contributions will be licensed under the Apache License 2.0.

---

*Copyright 2026 ResilientMind AI | ResilientMindai.com | Joseph C McGinty Jr*
