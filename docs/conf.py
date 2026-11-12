# Sphinx configuration for the pymsz documentation.
#
# Build locally (from the repository root):
#     pip install -r docs/requirements.txt
#     make -C docs html            # or: sphinx-build -b html docs docs/_build/html
# and open docs/_build/html/index.html.
#
# The pages are Markdown (MyST). The API reference is generated from the docstrings
# (NumPy style) with autodoc/autosummary/napoleon. pymsz is imported from this source tree; it is not
# installed and the SZpack C++ extension is not compiled for the documentation build.

import os
import re
import sys
import types
import datetime

DOCS = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(DOCS)
sys.path.insert(0, ROOT)


# -- version ------------------------------------------------------------------------------------
def _version_from_setup():
    with open(os.path.join(ROOT, 'setup.py')) as f:
        txt = f.read()
    major = re.search(r'^MAJOR\s*=\s*(\d+)', txt, re.M)
    minor = re.search(r'^MINOR\s*=\s*(\d+)', txt, re.M)
    return '%s.%s' % (major.group(1), minor.group(1)) if major and minor else 'unknown'


release = version = _version_from_setup()

# pymsz/version.py is written by setup.py and is not in git. On a fresh checkout (e.g. on Read the
# Docs) provide it in memory so that `import pymsz` works without installing the package.
if not os.path.exists(os.path.join(ROOT, 'pymsz', 'version.py')):
    _v = types.ModuleType('pymsz.version')
    _v.version = release
    _v.git_revision = 'unknown'
    sys.modules['pymsz.version'] = _v

# -- project ------------------------------------------------------------------------------------
project = 'pymsz'
author = 'Weiguang Cui'
copyright = '2017-%d, Weiguang Cui and contributors' % datetime.date.today().year

# -- general ------------------------------------------------------------------------------------
extensions = [
    'myst_nb',                     # Markdown (MyST) pages and Jupyter notebooks
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.napoleon',         # NumPy-style docstrings
    'sphinx.ext.viewcode',
    'sphinx.ext.intersphinx',
    'sphinx.ext.mathjax',
    'sphinx_copybutton',
]

source_suffix = {'.rst': 'restructuredtext', '.md': 'myst-nb', '.ipynb': 'myst-nb'}
root_doc = 'index'
# reference/_generated holds tables included by other pages, not pages of their own
exclude_patterns = ['_build', 'reference/_generated', '**.ipynb_checkpoints', 'Thumbs.db', '.DS_Store']
templates_path = ['_templates']

# `name` in docstrings: try to link to a Python object, else render as code
default_role = 'py:obj'

# -- MyST / notebooks ---------------------------------------------------------------------------
myst_enable_extensions = ['colon_fence', 'dollarmath', 'amsmath', 'deflist', 'fieldlist', 'attrs_inline',
                          'substitution']
myst_heading_anchors = 3
myst_substitutions = {'version': release}
# Notebooks are rendered with the outputs stored in them (they need the test snapshot and the
# compiled SZpack extension to run).
nb_execution_mode = 'off'

# -- autodoc / autosummary / napoleon -----------------------------------------------------------
autosummary_generate = True
autosummary_imported_members = False
autodoc_default_options = {
    'members': True,
    'member-order': 'bysource',
    'undoc-members': False,
    'show-inheritance': False,
}
autoclass_content = 'class'          # the parameters are documented in the class docstrings
autodoc_typehints = 'none'
autodoc_preserve_defaults = True     # show `SP=[194.95, 27.98]`, not the evaluated object

napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_use_param = True
napoleon_use_rtype = False
napoleon_use_ivar = True
napoleon_preprocess_types = False

# -- cross references ---------------------------------------------------------------------------
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'scipy': ('https://docs.scipy.org/doc/scipy/', None),
    'astropy': ('https://docs.astropy.org/en/stable/', None),
}

# -- HTML ---------------------------------------------------------------------------------------
html_theme = 'furo'
html_title = 'pymsz %s' % release
html_theme_options = {
    'source_repository': 'https://github.com/weiguangcui/pymsz/',
    'source_branch': 'master',
    'source_directory': 'docs/',
}
copybutton_prompt_text = r'>>> |\.\.\. |\$ '
copybutton_prompt_is_regexp = True


# -- generated reference tables -----------------------------------------------------------------
def _write_if_changed(path, text):
    if os.path.exists(path):
        with open(path) as f:
            if f.read() == text:
                return
    with open(path, 'w') as f:
        f.write(text)


def _generate_tables():
    """Tables of the telescope presets and the shipped bandpasses, read from the code so that the
    documentation always matches it. Written to docs/reference/_generated/ (not in git)."""
    from pymsz.telescope import PRESETS
    from pymsz import bandpasses

    out = os.path.join(DOCS, 'reference', '_generated')
    os.makedirs(out, exist_ok=True)

    def num(v, fmt='%.4g'):
        return '' if v is None else fmt % v

    L = []
    for key in sorted(PRESETS, key=str.lower):
        p = PRESETS[key]
        if key == 'CCAT':  # alias of FYST
            continue
        L.append('## %s\n' % key)
        L.append('%s. Map pixel: %s arcmin.\n' % (p.get('description', ''), num(p.get('pixel'))))
        L.append('| Band | Frequency (GHz) | Beam FWHM (arcmin) | White noise | 1/f knee $\\ell_\\mathrm{knee}$ '
                 '| Calibration error | Bandpass (default) |')
        L.append('|---|---|---|---|---|---|---|')
        for b in p['bands']:
            if b.get('bandpass'):
                bp = '`%s`' % b['bandpass']
            elif b.get('bandwidth'):
                bp = 'top-hat, %g GHz wide' % b['bandwidth']
            elif b.get('frac_bandwidth'):
                bp = 'top-hat, %g%% wide' % (100 * b['frac_bandwidth'])
            else:
                bp = 'monochromatic'
            L.append('| `%s` | %g | %.3g | %.3g %s | %s | %s | %s |' % (
                b['name'], b['freq'], b.get('fwhm', 0.), b.get('noise', 0.), b.get('noise_unit', 'uK-arcmin'),
                num(b.get('ell_knee'), '%.0f') or '—', num(b.get('calib_error'), '%.2g') or '—', bp))
        L.append('\n**Sources:** %s\n' % p.get('reference', ''))
    L.append('`CCAT` is an alias of `FYST`.\n')
    _write_if_changed(os.path.join(out, 'presets.md'), '\n'.join(L) + '\n')

    groups = {}
    for name in bandpasses.available():
        groups.setdefault(name.split('/')[0], []).append(name)
    B = ['| Instrument | Names for `bandpasses.load()` and `Band(bandpass=...)` |', '|---|---|']
    for g in sorted(groups):
        B.append('| %s | %s |' % (g, ', '.join('`%s`' % n for n in groups[g])))
    _write_if_changed(os.path.join(out, 'bandpass_names.md'), '\n'.join(B) + '\n')


_generate_tables()
