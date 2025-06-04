#!/usr/bin/env bash
set -e

# Simple script to set up the pipenv environment for PyWarp

# Install pipenv if it's not already available
if ! command -v pipenv >/dev/null 2>&1; then
    echo "pipenv not found, installing..."
    pip install --user pipenv
fi

# Install the project dependencies defined in the Pipfile
pipenv install --dev

cat <<MSG

Environment is ready. Activate it with:
  pipenv shell
MSG

