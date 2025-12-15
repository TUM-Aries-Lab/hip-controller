# Template-Python

Do ***NOT*** clone this repository. Please use it as a template instead—this README is meant to help you get started quickly.

The file cookiecutter.json has the following contents:
```
{
  "repo_name": "new-repo",
  "module_name": "new_repo",
  "package_name": "{{ cookiecutter.repo_name }}",
  "org_name": "TUM-Aries-Lab",
  "description": "Basic description of the repo.",
  "author_name": "First Last",
  "author_email": "first.last@tum.de",
  "python_version": "3.12",
  "version": "0.0.1alpha"
}
```

## Steps
1. Change premissions `chmod +x setup.sh`
2. Run the bash script: `./setup.sh`
3. Enter in your desired field info
4. Delete the `setup.sh` file
