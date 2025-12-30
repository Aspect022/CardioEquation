# Contributing to CardioEquation

First off, thank you for considering contributing to CardioEquation! It's people like you that make CardioEquation such a great tool for advancing personalized cardiac analysis.

## 🫀 Welcome!

CardioEquation is an open-source project that aims to revolutionize personalized ECG analysis through AI-driven mathematical modeling. We welcome contributions from developers, researchers, clinicians, and anyone interested in improving cardiac care through technology.

## 📋 Table of Contents

- [Code of Conduct](#code-of-conduct)
- [How Can I Contribute?](#how-can-i-contribute)
- [Getting Started](#getting-started)
- [Development Process](#development-process)
- [Coding Standards](#coding-standards)
- [Testing Guidelines](#testing-guidelines)
- [Documentation](#documentation)
- [Submitting Changes](#submitting-changes)
- [Community](#community)

## 📜 Code of Conduct

This project and everyone participating in it is governed by our [Code of Conduct](CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code. Please report unacceptable behavior to the project maintainers.

## 🤝 How Can I Contribute?

### Reporting Bugs

Before creating bug reports, please check the existing issues to avoid duplicates. When you create a bug report, include as many details as possible:

**Bug Report Template:**
- **Description**: Clear and concise description of the bug
- **Steps to Reproduce**: Numbered steps to reproduce the behavior
- **Expected Behavior**: What you expected to happen
- **Actual Behavior**: What actually happened
- **Environment**: 
  - OS (e.g., Ubuntu 20.04, Windows 11, macOS 13)
  - Python version
  - TensorFlow version
  - Relevant package versions
- **Additional Context**: Screenshots, error logs, or other relevant information

### Suggesting Enhancements

Enhancement suggestions are tracked as GitHub issues. When creating an enhancement suggestion, include:

- **Clear title** describing the enhancement
- **Detailed description** of the proposed functionality
- **Use case**: Why would this enhancement be useful?
- **Potential implementation**: If you have ideas on how to implement it
- **Alternatives considered**: What other solutions did you consider?

### Your First Code Contribution

Unsure where to begin? Look for issues labeled:
- `good first issue` - Simple issues perfect for newcomers
- `help wanted` - Issues where we'd love community input
- `documentation` - Improvements to docs
- `bug` - Known bugs that need fixing

### Pull Requests

1. Fork the repository
2. Create a new branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Write or update tests
5. Update documentation
6. Commit your changes (`git commit -m 'Add amazing feature'`)
7. Push to the branch (`git push origin feature/amazing-feature`)
8. Open a Pull Request

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- Git
- Basic understanding of ECG signals (for cardiac-related contributions)
- Familiarity with TensorFlow/Keras (for ML contributions)

### Setting Up Development Environment

```bash
# 1. Fork and clone the repository
git clone https://github.com/YOUR_USERNAME/CardioEquation.git
cd CardioEquation

# 2. Create a virtual environment
python -m venv venv

# On Windows:
venv\Scripts\activate

# On macOS/Linux:
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Install development dependencies (if available)
pip install pytest black flake8 mypy

# 5. Run tests to ensure everything works
python -m pytest tests/

# 6. Create a new branch for your work
git checkout -b feature/your-feature-name
```

## 🔄 Development Process

### Branch Naming Convention

- `feature/description` - New features
- `bugfix/description` - Bug fixes
- `docs/description` - Documentation updates
- `refactor/description` - Code refactoring
- `test/description` - Test additions/modifications

### Commit Messages

Follow the [Conventional Commits](https://www.conventionalcommits.org/) specification:

```
<type>(<scope>): <subject>

<body>

<footer>
```

**Types:**
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting, etc.)
- `refactor`: Code refactoring
- `test`: Adding or updating tests
- `chore`: Maintenance tasks

**Examples:**
```
feat(generator): add support for 12-lead ECG generation

Added capability to generate 12-lead ECG signals with proper
lead configuration and spatial relationships.

Closes #123
```

```
fix(trainer): resolve NaN loss during training

Fixed numerical instability in parameter normalization that
caused NaN values during model training.

Fixes #456
```

## 💻 Coding Standards

### Python Style Guide

We follow [PEP 8](https://pep8.org/) with some modifications:

- **Line Length**: Maximum 100 characters (instead of 79)
- **Indentation**: 4 spaces (no tabs)
- **Quotes**: Use single quotes for strings unless double quotes avoid escaping
- **Imports**: Group and sort alphabetically (stdlib, third-party, local)

### Code Formatting

We use `black` for code formatting:

```bash
# Format your code before committing
black src/ tests/

# Check formatting
black --check src/ tests/
```

### Type Hints

Use type hints for function signatures:

```python
def generate_ecg(
    params: dict[str, float], 
    num_beats: int = 5, 
    fs: int = 500
) -> np.ndarray:
    """Generate synthetic ECG signal."""
    ...
```

### Documentation Strings

Use Google-style docstrings:

```python
def example_function(param1: int, param2: str) -> bool:
    """
    Brief description of the function.

    More detailed description if needed. Explain the purpose,
    behavior, and any important considerations.

    Args:
        param1: Description of param1
        param2: Description of param2

    Returns:
        Description of return value

    Raises:
        ValueError: When param1 is negative
        TypeError: When param2 is not a string

    Examples:
        >>> example_function(5, "test")
        True
    """
    ...
```

### Naming Conventions

- **Variables/Functions**: `snake_case`
- **Classes**: `PascalCase`
- **Constants**: `UPPER_SNAKE_CASE`
- **Private methods**: `_leading_underscore`
- **"Magic" methods**: `__double_leading_underscore__`

## 🧪 Testing Guidelines

### Writing Tests

- Write tests for all new functionality
- Maintain or improve code coverage
- Use descriptive test names: `test_<function>_<scenario>_<expected_result>`
- Group related tests in classes

```python
import pytest
from src.ecg_generator import generate_ecg

class TestECGGeneration:
    """Tests for ECG generation functionality."""
    
    def test_generate_ecg_returns_correct_length(self):
        """Test that generated ECG has expected length."""
        params = {...}
        ecg = generate_ecg(params, num_beats=5, fs=500)
        expected_length = 5 * (60 / params['HR']) * 500
        assert len(ecg) == pytest.approx(expected_length, rel=0.01)
    
    def test_generate_ecg_with_invalid_params_raises_error(self):
        """Test that invalid parameters raise appropriate errors."""
        with pytest.raises(ValueError):
            generate_ecg({'HR': -10})
```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src tests/

# Run specific test file
pytest tests/test_ecg_generator.py

# Run specific test
pytest tests/test_ecg_generator.py::TestECGGeneration::test_generate_ecg_returns_correct_length
```

### Test Coverage

- Aim for at least 80% code coverage
- Focus on testing critical paths and edge cases
- Don't sacrifice test quality for coverage percentage

## 📚 Documentation

### Code Documentation

- Write clear, concise docstrings for all public functions, classes, and modules
- Include examples in docstrings where helpful
- Keep docstrings up-to-date with code changes

### Project Documentation

When adding new features:
1. Update relevant sections in README.md
2. Add or update documentation in `docs/` directory
3. Create usage examples if appropriate
4. Update CHANGELOG.md

### Documentation Format

- Use Markdown for all documentation
- Include code examples with syntax highlighting
- Add diagrams or images where they improve understanding
- Keep language clear and accessible

## 📤 Submitting Changes

### Pull Request Process

1. **Before submitting:**
   - Run tests: `pytest`
   - Format code: `black src/ tests/`
   - Check linting: `flake8 src/ tests/`
   - Update documentation
   - Update CHANGELOG.md

2. **Pull Request Description:**
   - Use the PR template
   - Reference related issues
   - Describe changes clearly
   - Include screenshots for UI changes
   - List any breaking changes

3. **Review Process:**
   - Maintainers will review your PR
   - Address review comments promptly
   - Be open to feedback and suggestions
   - Keep PR scope focused

4. **After Approval:**
   - Maintainer will merge your PR
   - Your contribution will be credited in release notes

### Pull Request Checklist

- [ ] Code follows project style guidelines
- [ ] All tests pass
- [ ] New tests added for new functionality
- [ ] Documentation updated
- [ ] CHANGELOG.md updated
- [ ] Commits follow conventional commit format
- [ ] No merge conflicts with main branch
- [ ] PR description is clear and complete

## 🎯 Areas for Contribution

### High Priority

- **Real ECG Integration**: PhysioNet dataset integration
- **Clinical Validation**: Testing with real patient data
- **Performance Optimization**: GPU acceleration, model compression
- **Multi-lead Support**: Extending to 12-lead ECG

### Feature Development

- **Symbolic Equation Generation**: Convert parameters to readable equations
- **Interactive Dashboard**: Streamlit/Dash visualization interface
- **Anomaly Detection**: Identify abnormal ECG patterns
- **Mobile Integration**: Support for wearable devices

### Research & Analysis

- **Pathology Modeling**: Disease-specific parameter sets
- **Drug Response Simulation**: Model medication effects
- **Longitudinal Analysis**: Track cardiac health over time
- **Biometric Applications**: Cardiac-based authentication

### Documentation & Community

- **Tutorial Creation**: Step-by-step guides
- **Example Notebooks**: Jupyter notebooks demonstrating use cases
- **Video Tutorials**: Screencasts and presentations
- **Translation**: Documentation in other languages

## 🌟 Recognition

Contributors are recognized in:
- README.md contributors section
- Release notes
- CHANGELOG.md
- Git commit history

Significant contributors may be invited to join the core team!

## 💬 Community

### Getting Help

- **GitHub Issues**: For bugs and feature requests
- **Discussions**: For questions, ideas, and general discussion
- **Email**: For sensitive matters or private inquiries

### Code Review

All submissions require review. We use GitHub pull requests for this purpose:
- Reviews focus on code quality, correctness, and maintainability
- Reviewers should be constructive and respectful
- Authors should be receptive to feedback

### Communication Guidelines

- **Be respectful**: Treat others with respect and professionalism
- **Be clear**: Communicate clearly and concisely
- **Be patient**: Remember that maintainers and reviewers are often volunteers
- **Be collaborative**: Work together to find the best solutions

## 🙏 Thank You!

Your contributions, whether code, documentation, bug reports, or feature suggestions, help make CardioEquation better for everyone. We appreciate your time and effort!

---

**Questions?** Feel free to reach out by opening a discussion or issue on GitHub.

**Ready to contribute?** Check out our [good first issues](https://github.com/Aspect022/CardioEquation/labels/good%20first%20issue) to get started!
