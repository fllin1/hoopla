import subprocess
import sys


def main() -> None:
    """Run tests for Gemini API integration."""
    result = subprocess.run(
        ["python", "-m", "pytest", "tests/test_gemini.py", "-s"],
        cwd=".",
    )
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
