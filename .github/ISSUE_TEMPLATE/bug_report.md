---
name: Bug Report
about: Create a report to help us improve CardioEquation
title: '[BUG] '
labels: bug
assignees: ''
---

## 🐛 Bug Description

**Clear and concise description of the bug**

A clear and concise description of what the bug is.

## 📋 Steps to Reproduce

**Steps to reproduce the behavior:**

1. Go to '...'
2. Run command '....'
3. Use parameters '....'
4. See error

## ✅ Expected Behavior

**What you expected to happen**

A clear and concise description of what you expected to happen.

## ❌ Actual Behavior

**What actually happened**

A clear and concise description of what actually happened.

## 🖼️ Screenshots / Error Messages

**If applicable, add screenshots or error messages**

```
Paste error messages or logs here
```

## 💻 Environment

**Please complete the following information:**

- **OS**: [e.g., Ubuntu 20.04, Windows 11, macOS 13]
- **Python Version**: [e.g., 3.8.10, 3.10.5]
- **CardioEquation Version**: [e.g., 1.0.0]
- **TensorFlow Version**: [e.g., 2.12.0]
- **Installation Method**: [e.g., pip, conda, from source]

**Dependencies:**
```bash
# Output of: pip list | grep -E "tensorflow|numpy|scipy|scikit-learn"
```

## 🔍 Additional Context

**Add any other context about the problem here**

- Have you made any modifications to the code?
- Does this happen consistently or intermittently?
- Did this work in a previous version?
- Any other relevant information

## 🧪 Reproducible Example (Optional)

**Minimal code to reproduce the issue**

```python
# Your code here that reproduces the bug
import numpy as np
from src.ecg_generator import generate_ecg

params = {...}
ecg = generate_ecg(params)  # Error occurs here
```

## 🔗 Related Issues

**Link to related issues (if any)**

- Related to #issue_number
- Similar to #issue_number

---

**Checklist before submitting:**

- [ ] I have searched existing issues to avoid duplicates
- [ ] I have provided a clear description of the bug
- [ ] I have included steps to reproduce
- [ ] I have provided environment information
- [ ] I have included error messages or logs (if applicable)
