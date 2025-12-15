#!/usr/bin/env bash
set -euo pipefail

# Capture existing folders before running cookiecutter
before=($(ls -1d */ 2>/dev/null || true))

# 1. Run cookiecutter in the current directory
cookiecutter .

# Capture folders after cookiecutter
after=($(ls -1d */ 2>/dev/null || true))

# Find the new folder
REPO_NAME=""
for dir in "${after[@]}"; do
    skip=false
    for old in "${before[@]}"; do
        if [[ "$dir" == "$old" ]]; then
            skip=true
            break
        fi
    done
    if [[ "$skip" == false ]]; then
        REPO_NAME="${dir%/}"
        break
    fi
done

if [[ -z "$REPO_NAME" ]]; then
  echo "❌ Error: Could not determine generated project folder."
  exit 1
fi

echo "✔ Detected generated project folder: $REPO_NAME"

# 2. Remove the template source folder safely
if [[ -d "{{ cookiecutter.repo_name }}" ]]; then
    echo "Deleting template folder '{{ cookiecutter.repo_name }}'"
    rm -rf "{{ cookiecutter.repo_name }}"
fi

# 3. Move files out of generated folder
echo "Moving files from $REPO_NAME to project root..."

shopt -s dotglob nullglob
for f in "$REPO_NAME"/*; do
    [[ "$f" == "$REPO_NAME/."  ]] && continue
    [[ "$f" == "$REPO_NAME/.." ]] && continue
    mv "$f" .
done
shopt -u dotglob nullglob

# 4. Remove now-empty generated folder
echo "Deleting inner folder: $REPO_NAME"
rmdir "$REPO_NAME"

# 5. Remove cookiecutter.json
echo "Deleting cookiecutter.json"
rm -f cookiecutter.json

echo "🎉 Setup complete!"
