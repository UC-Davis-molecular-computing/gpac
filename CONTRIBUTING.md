# Contributing to gpac

gpac is a plain (pure-Python) library, so there is not much process. This document explains how to set up a development environment, the few conventions we follow, and — mainly — how to publish a new release.

## Development setup

Clone the repo and install the package in editable mode along with its dependencies:

```
git clone https://github.com/UC-Davis-molecular-computing/gpac.git
cd gpac
pip install -e .
```

There is currently no automated test suite; [notebook.ipynb](notebook.ipynb) exercises most of the library and doubles as the example documentation, so after making changes it is a good idea to re-run it and check that the examples still work.

## Code style

Code is formatted/linted with [Ruff](https://docs.astral.sh/ruff/) using a line length of 120 (configured in [ruff.toml](ruff.toml)):

```
ruff check gpac
ruff format gpac
```

Docstrings use [NumPy style](https://numpydoc.readthedocs.io/en/latest/format.html) and are rendered by mkdocstrings, so keep them well-formed — they are the API documentation.

## Documentation

API docs are built with [MkDocs](https://www.mkdocs.org/) + [mkdocstrings](https://mkdocstrings.github.io/) and hosted at <https://gpac.readthedocs.io>. Read the Docs rebuilds them automatically on every push to `main`; nothing needs to be done manually. To preview locally:

```
pip install -e ".[docs]"
mkdocs serve
```

## Making a release

Publishing to PyPI is automated by the GitHub Actions workflow
[.github/workflows/python-publish.yml](.github/workflows/python-publish.yml):
creating a GitHub Release builds the sdist + wheel and uploads them to PyPI
(via PyPI "trusted publishing"; no tokens involved). The scripts in
[scripts/](scripts/) automate creating that release safely. You need the
[GitHub CLI](https://cli.github.com/) (`gh`), authenticated with
`gh auth login`, and push access to the repo.

Steps:

1. **Bump the version** in [pyproject.toml](pyproject.toml) (the `version = "X.Y.Z"` line near the top). Use [semantic versioning](https://semver.org/): patch for bug fixes, minor for new backwards-compatible features, major for breaking changes. This is the only place the version number lives.

2. **Commit and push to `main`** (or merge your PR). The release script refuses to run unless your local `main` exactly matches `origin/main`, so the release is exactly what you tested.

3. **Run the release script** from the repo root:

   ```
   ./scripts/release.ps1      # Windows (PowerShell 7+, i.e., pwsh)
   bash scripts/release.sh    # macOS / Linux
   ```

   The script:
   - checks that the version in pyproject.toml is strictly newer than the latest GitHub release, and that the tag `vX.Y.Z` doesn't already exist;
   - checks you are on `main` and in sync with `origin/main`;
   - asks you to type the tag (e.g., `v1.2.3`) to confirm — **this step publishes irreversibly**: a version number can never be re-uploaded to PyPI once published;
   - creates the GitHub Release (which triggers the build + publish workflow) and watches the workflow until it finishes.

4. **Verify** at `https://pypi.org/project/gpac/` that the new version is up. Release notes are auto-generated on the [releases page](https://github.com/UC-Davis-molecular-computing/gpac/releases); edit them there if you want to say more.

### If the release fails

If the workflow fails partway (e.g., a build error), fix the problem, then delete **both** the GitHub release **and** its tag before re-running the script with the same version:

```
gh release delete vX.Y.Z --yes --cleanup-tag
```

The release script refuses to run while the tag still exists on origin, so a leftover tag can't cause a bad build through the script — but if you instead re-create the release by hand (`gh release create` or the web UI) on a leftover tag, the workflow builds the files from the commit the tag points to, not the latest commit. If the failure happened *after* the PyPI upload succeeded, you must instead bump to a new version number — PyPI does not allow re-uploading a version, even a deleted one.

### Testing the build without publishing

To check that the package builds in CI without creating a release or uploading anything:

```
./scripts/test-build.ps1      # Windows (PowerShell 7+, i.e., pwsh)
bash scripts/test-build.sh    # macOS / Linux
```

This dispatches the same workflow manually; the publish job is skipped for manual runs. You can pass a branch name as an argument to test a branch other than `main`.

### Manual release (fallback)

If GitHub Actions is unavailable, you can build and upload by hand from the repo root (requires a PyPI API token configured for `twine`):

```
python -m pip install build twine
python -m build
twine upload dist/*
```
