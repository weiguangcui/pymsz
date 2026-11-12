# Getting help and contributing

## Reporting problems

Please report bugs and ask questions on the
[GitHub issue tracker](https://github.com/weiguangcui/pymsz/issues), or write to Weiguang Cui
(cuiweiguang@gmail.com). Include the pymsz version (`pymsz.version.version`), the code or parameter
file you ran, and the full error message.

## Contributing

Contributions are welcome as pull requests on
[GitHub](https://github.com/weiguangcui/pymsz/pulls).

- Install pymsz in development mode with the test tools: `pip install -e ".[yaml,test]"`.
- Add tests for new features and run `pytest` before opening a pull request. GitHub runs the tests
  again on the pull request. See {doc}`testing`.
- Document new functions and classes with NumPy-style docstrings (sections `Parameters`, `Returns`,
  `Notes`, `Examples`). They appear automatically in the {doc}`api/index`.
- Add new user-facing features to the {doc}`user guide <user_guide/pipeline>` pages in `docs/`.

## Building the documentation

The documentation is written in Markdown ([MyST](https://myst-parser.readthedocs.io/)) and built
with [Sphinx](https://www.sphinx-doc.org/). The API pages are generated from the docstrings.

```bash
pip install -r docs/requirements.txt
make -C docs html        # open docs/_build/html/index.html
make -C docs clean       # remove the build and the generated pages
```

| Path | Content |
|---|---|
| `docs/conf.py` | Sphinx configuration |
| `docs/*.md`, `docs/user_guide/`, `docs/reference/` | the pages |
| `docs/tutorials/*.ipynb` | tutorial notebooks, shown with their stored outputs (they are not run during the build) |
| `docs/api/` | API reference; `docs/api/generated/` is created by the build |
| `.readthedocs.yaml` | configuration of [Read the Docs](https://readthedocs.org/), which publishes the documentation |

To update a tutorial, run the notebook in Jupyter from `docs/tutorials/` (it needs the SZpack
extension and the test snapshot) and save it with its outputs.

## Releasing a new version

pymsz is published on [PyPI](https://pypi.org/project/pymsz/) by the GitHub workflow
`.github/workflows/release.yml`. It builds the source package and the binary wheels (Linux x86_64 and
aarch64, macOS arm64; Python 3.10-3.14; settings in `pyproject.toml`, `[tool.cibuildwheel]`), tests
every wheel, and uploads them. No password or token is stored: PyPI trusts this workflow
(*trusted publishing*).

### One-time setup

1. Create accounts on [pypi.org](https://pypi.org/account/register/) and
   [test.pypi.org](https://test.pypi.org/account/register/) (two separate accounts; both require
   two-factor authentication).
2. On each site, go to **Your account → Publishing → Add a new pending publisher** and enter:

   | Field | pypi.org | test.pypi.org |
   |---|---|---|
   | PyPI project name | `pymsz` | `pymsz` |
   | Owner | `weiguangcui` | `weiguangcui` |
   | Repository name | `pymsz` | `pymsz` |
   | Workflow name | `release.yml` | `release.yml` |
   | Environment name | `pypi` | `testpypi` |

   The first upload then creates the project and makes you its owner.
3. On GitHub, **Settings → Environments → New environment**: create `pypi` and `testpypi`. For
   `pypi`, you can add yourself under *Required reviewers*, so that every release waits for your
   approval.

### Every release

1. Set the version in `setup.py` (`MAJOR`, `MINOR`), add the changes to `docs/changelog.md`, commit
   and push. Wait for the tests to pass.
2. Trial run: **Actions → release → Run workflow**. It builds everything and uploads it to TestPyPI.
   Check it in a fresh environment:

   ```bash
   python -m venv /tmp/t && source /tmp/t/bin/activate
   pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ "pymsz[yaml]"
   python -c "import pymsz; from pymsz import szpack; print(pymsz.version.version, szpack.available)"
   ```
3. Publish: tag the commit with the version and push the tag.

   ```bash
   git tag v1.0
   git push origin v1.0
   ```

   The workflow checks that the tag matches the version in `setup.py`, builds and tests everything
   again, and publishes it on PyPI. A version can be uploaded to PyPI only once: to fix a broken
   release, publish a new version (e.g. 1.1).

On Read the Docs, activate the new tag under **Versions** (or add an automation rule that activates
new tags), so that the documentation of each release stays available next to `latest`.
