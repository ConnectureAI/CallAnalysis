# Contributing to Call Analysis System

Thank you for your interest in contributing to the Call Analysis System! This document provides guidelines and information for contributors.

## 🎯 How to Contribute

### Types of Contributions

We welcome several types of contributions:

- **Bug reports** - Help us identify and fix issues
- **Feature requests** - Suggest new functionality or improvements
- **Code contributions** - Implement features, fix bugs, or improve performance
- **Documentation** - Improve docs, add examples, or write tutorials
- **Testing** - Add tests, improve test coverage, or test new features

### Before You Start

1. **Check existing issues** - Look for existing issues or discussions about your idea
2. **Create an issue** - For significant changes, create an issue to discuss your approach
3. **Fork the repository** - Create your own fork to work on

## 🚀 Development Setup

### Prerequisites

- Python 3.9+
- Docker and Docker Compose
- Git
- OpenAI API key (for testing AI features)

### Initial Setup

```bash
# Clone your fork
git clone https://github.com/yourusername/call-analysis.git
cd call-analysis

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install development dependencies
pip install -e ".[dev]"

# Copy environment template
cp .env.example .env
# Edit .env with your test configuration

# Start development services
./scripts/start-dev.sh
```

### Development Workflow

1. **Create a branch** for your feature/fix:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes** following our coding standards

3. **Run tests** to ensure everything works:
   ```bash
   pytest
   ```

4. **Run code quality checks**:
   ```bash
   # Format code
   black src/ tests/
   
   # Check linting
   ruff check src/ tests/
   
   # Type checking
   mypy src/
   ```

5. **Commit your changes**:
   ```bash
   git add .
   git commit -m "Add feature: your feature description"
   ```

6. **Push to your fork**:
   ```bash
   git push origin feature/your-feature-name
   ```

7. **Create a Pull Request** on GitHub

## 📝 Coding Standards

### Code Style

We use these tools to maintain code quality:

- **Black** for code formatting
- **Ruff** for linting
- **MyPy** for type checking
- **isort** for import sorting

### Code Guidelines

#### Python Code Style
- Follow PEP 8 style guidelines
- Use type hints for all functions and methods
- Write comprehensive docstrings using Google style
- Keep functions focused and small (< 50 lines when possible)
- Use meaningful variable and function names

#### Example Function
```python
async def analyze_transcript(
    self, 
    transcript: str, 
    call_id: str,
    options: Optional[AnalysisOptions] = None
) -> CallInsight:
    """
    Analyze a call transcript and extract structured insights.
    
    Args:
        transcript: The call transcript text to analyze
        call_id: Unique identifier for the call
        options: Optional analysis configuration
        
    Returns:
        CallInsight object containing extracted information
        
    Raises:
        AnalysisError: If transcript analysis fails
        ValidationError: If input parameters are invalid
    """
    # Implementation here
```

#### Database Models
- Use descriptive table and column names
- Add proper indexes for query performance
- Include created_at and updated_at timestamps
- Use appropriate data types and constraints

#### API Design
- Follow RESTful conventions
- Use proper HTTP status codes
- Include comprehensive error handling
- Provide clear API documentation
- Use Pydantic models for request/response validation

### Testing Guidelines

#### Test Structure
```python
# tests/test_analyzer.py
import pytest
from unittest.mock import AsyncMock, patch

from src.call_analysis.analyzer import SemanticTranscriptAnalyzer
from src.call_analysis.models import CallInsight


class TestSemanticTranscriptAnalyzer:
    """Test suite for SemanticTranscriptAnalyzer."""
    
    @pytest.fixture
    def analyzer(self):
        """Create analyzer instance for testing."""
        return SemanticTranscriptAnalyzer(api_key="test-key")
    
    @pytest.mark.asyncio
    async def test_analyze_transcript_success(self, analyzer):
        """Test successful transcript analysis."""
        # Test implementation
        pass
    
    @pytest.mark.asyncio
    async def test_analyze_transcript_api_error(self, analyzer):
        """Test handling of API errors."""
        # Test implementation
        pass
```

#### Test Categories
- **Unit tests** - Test individual functions/methods in isolation
- **Integration tests** - Test component interactions
- **API tests** - Test HTTP endpoints
- **Performance tests** - Test system performance under load

#### Test Requirements
- Maintain test coverage above 80%
- Test both success and failure scenarios
- Use meaningful test names that describe what's being tested
- Mock external dependencies (OpenAI API, database, etc.)
- Include edge cases and boundary conditions

## 🐛 Bug Reports

When reporting bugs, please include:

### Bug Report Template
```markdown
## Bug Description
A clear description of what the bug is.

## Steps to Reproduce
1. Go to '...'
2. Click on '....'
3. Scroll down to '....'
4. See error

## Expected Behavior
What you expected to happen.

## Actual Behavior
What actually happened.

## Environment
- OS: [e.g., Ubuntu 20.04, Windows 10, macOS 12]
- Python version: [e.g., 3.9.7]
- Package version: [e.g., 0.1.0]
- Docker version (if applicable): [e.g., 20.10.8]

## Additional Context
Any other context about the problem here.

## Logs
```
Paste relevant logs here
```
```

## 💡 Feature Requests

When suggesting features, please include:

### Feature Request Template
```markdown
## Feature Description
A clear description of the feature you'd like to see.

## Problem Statement
What problem does this feature solve?

## Proposed Solution
Describe your proposed solution.

## Alternative Solutions
Describe any alternative solutions you've considered.

## Additional Context
Any other context, screenshots, or examples.

## Implementation Ideas
If you have ideas about how to implement this feature.
```

## 🔄 Pull Request Process

### Pull Request Checklist

Before submitting a PR, ensure:

- [ ] Code follows project style guidelines
- [ ] Tests pass locally (`pytest`)
- [ ] Code coverage is maintained/improved
- [ ] Documentation is updated if needed
- [ ] Commit messages are clear and descriptive
- [ ] PR description explains the changes
- [ ] No sensitive information is included

### Pull Request Template
```markdown
## Description
Brief description of the changes.

## Type of Change
- [ ] Bug fix (non-breaking change which fixes an issue)
- [ ] New feature (non-breaking change which adds functionality)
- [ ] Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] Documentation update

## Testing
- [ ] Unit tests added/updated
- [ ] Integration tests added/updated
- [ ] Manual testing completed

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Code is commented where necessary
- [ ] Documentation updated
- [ ] Tests pass
- [ ] No merge conflicts
```

### Review Process

1. **Automated checks** - CI/CD pipeline runs tests and quality checks
2. **Code review** - Maintainers review code for quality and design
3. **Testing** - Changes are tested in a staging environment
4. **Approval** - At least one maintainer approves the changes
5. **Merge** - Changes are merged into the main branch

## 📚 Documentation

### Documentation Types

- **API Documentation** - Automatically generated from code
- **User Guides** - How to use the system
- **Developer Guides** - How to extend/modify the system
- **Deployment Guides** - How to deploy the system

### Writing Documentation

- Use clear, simple language
- Include code examples
- Add screenshots/diagrams where helpful
- Keep documentation up-to-date with code changes
- Test documentation steps to ensure they work

## 🌍 Community Guidelines

### Code of Conduct

We are committed to providing a welcoming and inspiring community for all. Please read our full [Code of Conduct](CODE_OF_CONDUCT.md).

### Communication

- **GitHub Issues** - For bug reports and feature requests
- **GitHub Discussions** - For questions and community discussions
- **Pull Requests** - For code contributions

### Be Respectful

- Use welcoming and inclusive language
- Be respectful of differing viewpoints and experiences
- Gracefully accept constructive criticism
- Focus on what is best for the community
- Show empathy towards other community members

## 🏷️ Release Process

### Version Numbering

We follow [Semantic Versioning](https://semver.org/):
- **MAJOR** version for incompatible API changes
- **MINOR** version for backwards-compatible functionality additions
- **PATCH** version for backwards-compatible bug fixes

### Release Workflow

1. **Development** - Features developed in feature branches
2. **Testing** - Comprehensive testing in staging environment
3. **Documentation** - Update documentation for new features
4. **Version Bump** - Update version numbers
5. **Release Notes** - Prepare detailed release notes
6. **Tag Release** - Create Git tag and GitHub release
7. **Deploy** - Deploy to production environment

## ❓ Getting Help

If you need help contributing:

1. **Check the documentation** - Look for existing guides
2. **Search issues** - Someone might have asked the same question
3. **Create a discussion** - Use GitHub Discussions for questions
4. **Contact maintainers** - Reach out to project maintainers

## 🎉 Recognition

Contributors are recognized in several ways:

- **Contributors list** - Listed in README.md
- **Release notes** - Mentioned in relevant release notes
- **GitHub contributors** - Automatically tracked by GitHub

Thank you for contributing to the Call Analysis System! 🚀