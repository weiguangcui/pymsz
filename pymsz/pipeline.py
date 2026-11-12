r"""
Mock SZ observations driven by a parameter file: simulation snapshot -> SZ model -> telescope maps.

One call runs the whole chain of pymsz::

    load_data           gas particles around every object of the snapshot(s)
      -> SZ model       for every redshift and line of sight: SZpack_model (relativistic tSZ + kSZ),
                        or TT_model + TK_model (y and b maps, with optional relativistic tSZ)
      -> observe        for every telescope: bandpasses, CMB, beam, filtering, pixels, gains, noise
      -> FITS files     observed maps and their components; optionally the model cubes

The parameter file is YAML (needs PyYAML), TOML or JSON. Write an annotated template, edit it, check
it and run it::

    pymsz-mock --template my_mock.yaml           # or: python -m pymsz --template my_mock.yaml
    pymsz-mock my_mock.yaml --dry-run             # check the parameters, print what will be done
    pymsz-mock my_mock.yaml                       # run
    pymsz-mock my_mock.yaml --set image.npixel=256 --set observation.seed=7    # override parameters

or from Python:

    >>> from pymsz import pipeline
    >>> recs = pipeline.run('my_mock.yaml')                      # one record (dict) per output map file
    >>> recs = pipeline.run(params_dict, keep_results=True)      # dict input; keep the MockObservations
    >>> recs = pipeline.run('my_mock.yaml', callback=f)          # f(obs, info) after every observation
    >>> pipe = pipeline.Pipeline('my_mock.yaml'); print(pipe.describe()); pipe.run()

Relative paths are relative to the directory of the parameter file (to the current directory for a
dict). The sections and their defaults are in DEFAULTS. Every parameter is documented in the template
pymsz/data/mock_params_template.yaml.
"""

import os
import sys
import copy
import json
import re
import time
import zlib
import shutil
import difflib
import inspect
import argparse
import numpy as np

from pymsz import bandpasses
from pymsz.telescope import Telescope, Band, PRESETS, cmb_power_spectrum

TEMPLATE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'mock_params_template.yaml')

DEFAULTS = {
    'simulation': dict(snapshot=None, name=None, center=None, radius=None, objects=None, catalogue=None,
                       restrict_r=True, cut_sfr=None, cut_rhoT=None, hmrad=None, metal=None, Nmets=11, mu=None),
    'image': dict(redshift=None, AR=None, npixel=500, axis='z', SP=[194.95, 27.98], sph_kernel='cubic', SD=2,
                  neighbours=None, zthick=None, Ncpu=None, Ntasks=None, Memreduce=False),
    'sz': dict(model='szpack', method='combo', projection='particle', kinetic=True, relativistic=True,
               bulk_velocity=None, szpack_options=None, Te_grid=None, frequencies=None, unit='MJy/sr'),
    'telescopes': [],
    'observation': dict(unit='uK_CMB', noise=True, cmb=None, foregrounds=None, calibration=True, seed=None,
                        pad=None),
    'output': dict(directory='pymsz_mock', name_format='{object}_z{redshift}_{projection}_{telescope}',
                   model_format='{object}_z{redshift}_{projection}_{model}', save_model=True, components=True,
                   overwrite=False, skip_existing=False, save_params=True),
}

_MODELS = {'szpack': 'szpack', 'tt+tk': 'tt+tk', 'tk+tt': 'tt+tk', 'tt': 'tt', 'tk': 'tk'}
_TEL_KEYS = ('preset', 'name', 'label', 'bands', 'band_options', 'measured_bandpass', 'pixel', 'transfer',
             'red_noise_corr', 'pointing_rms', 'description', 'reference')
_TEL_OBS_KEYS = ('unit', 'noise', 'cmb', 'foregrounds', 'calibration', 'pad')  # per-telescope overrides
_OBJ_KEYS = ('name', 'snapshot', 'center', 'radius') + tuple(DEFAULTS['image'])
_AXIS_KEYS = ('angles', 'matrix', 'random', 'seed', 'label')
_BAND_KEYS = tuple(inspect.signature(Band.__init__).parameters)[1:]


# ------------------------------------------------------------------------------------------------
# reading and checking the parameters
# ------------------------------------------------------------------------------------------------
_EXP_FLOAT = re.compile(r'^[-+]?(\d+\.?\d*|\.\d+)[eE][-+]?\d+$')


def _coerce(v):
    """Numbers that YAML 1.1 reads as strings (1e-4, 1.5e3) -> float; 'none'/'null' -> None (for TOML)."""
    if isinstance(v, dict):
        return {k: _coerce(x) for k, x in v.items()}
    if isinstance(v, list):
        return [_coerce(x) for x in v]
    if isinstance(v, str):
        s = v.strip()
        if _EXP_FLOAT.match(s):
            return float(s)
        if s.lower() in ('none', 'null', '~'):
            return None
    return v


def _yaml():
    try:
        import yaml
    except ImportError:
        raise ImportError("YAML parameter files need PyYAML (pip install pyyaml); or write the file in TOML "
                          "(.toml) or JSON (.json)")
    return yaml


def read_params(fname):
    """Read a parameter file (.yaml / .yml, .toml or .json) into a dict (not yet checked)."""
    ext = os.path.splitext(fname)[1].lower()
    with open(fname, 'rb') as f:
        text = f.read().decode('utf-8')
    if ext == '.toml':
        try:
            import tomllib
        except ImportError:
            try:
                import tomli as tomllib
            except ImportError:
                raise ImportError("TOML parameter files need Python >= 3.11 or tomli (pip install tomli)")
        p = tomllib.loads(text)
    elif ext == '.json':
        p = json.loads(text)
    else:
        p = _yaml().safe_load(text)
    if p is None:
        p = {}
    if not isinstance(p, dict):
        raise ValueError("%s: the parameter file must contain the sections %s" % (fname, ', '.join(DEFAULTS)))
    return _coerce(p)


def parse_value(text):
    """Value of a command-line override (YAML syntax if PyYAML is installed, else JSON, else a string)."""
    try:
        return _coerce(_yaml().safe_load(text))
    except ImportError:
        try:
            return json.loads(text)
        except ValueError:
            return _coerce(text)


def set_param(params, key, value):
    """params[a][b] = value for key 'a.b'. List items by index, e.g. 'telescopes.0.pixel'."""
    parts = key.split('.')
    d = params
    for i, k in enumerate(parts):
        last = i == len(parts) - 1
        if isinstance(d, list):
            j = int(k)
            if last:
                d[j] = value
                return
            if isinstance(d[j], str):  # 'SPT-3G' -> {'preset': 'SPT-3G'}
                d[j] = {'preset': d[j]}
            d = d[j]
        else:
            if last:
                d[k] = value
                return
            if d.get(k) is None:
                d[k] = {}
            d = d[k]


def _check_keys(d, allowed, where):
    if not isinstance(d, dict):
        raise ValueError("%s must be a mapping of parameters, got %r" % (where, d))
    for k in d:
        if k not in allowed:
            close = difflib.get_close_matches(str(k), list(allowed), 1)
            hint = " Did you mean '%s'?" % close[0] if close else ''
            raise ValueError("Unknown parameter '%s' in %s.%s Allowed: %s" % (k, where, hint, ', '.join(allowed)))


def _path(v, base):
    if v is None:
        return None
    v = os.path.expanduser(str(v))
    return os.path.normpath(v if os.path.isabs(v) else os.path.join(base, v))


def _band(d, base, where):
    """Check Band arguments and resolve the file names in them."""
    _check_keys(d, _BAND_KEYS, where)
    d = dict(d)
    bp = d.get('bandpass')
    if isinstance(bp, str) and not bandpasses.is_available(bp):
        f = _path(bp, base)
        if not os.path.isfile(f):
            close = difflib.get_close_matches(bp, bandpasses.available(), 3)
            raise ValueError("%s: bandpass '%s' is neither a bandpass shipped with pymsz%s nor a file (%s)"
                             % (where, bp, ' (did you mean %s?)' % ', '.join(close) if close else '', f))
        d['bandpass'] = f
    if isinstance(d.get('beam'), str):
        d['beam'] = _path(d['beam'], base)
    return d


def _is_angles(a):
    return isinstance(a, (list, tuple)) and len(a) == 3 and \
        all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in a)


def _telescope(t, base, i):
    where = 'telescopes[%d]' % i
    if isinstance(t, str):
        t = {'preset': t}
    _check_keys(t, _TEL_KEYS + _TEL_OBS_KEYS, where)
    t = dict(t)
    if 'preset' in t:
        keys = {k.lower(): k for k in PRESETS}
        if str(t['preset']).lower() not in keys:
            raise ValueError("%s: unknown preset '%s'. Presets: %s" % (where, t['preset'], ', '.join(sorted(PRESETS))))
        names = [b['name'] for b in PRESETS[keys[str(t['preset']).lower()]]['bands']]
        bo = t.get('band_options')
        if bo:
            if all(isinstance(v, dict) for v in bo.values()):
                for k in bo:
                    if k not in names:
                        raise ValueError("%s: band_options for band '%s', but the bands of %s are %s"
                                         % (where, k, t['preset'], ', '.join(names)))
                t['band_options'] = {k: _band(v, base, '%s.band_options.%s' % (where, k)) for k, v in bo.items()}
            else:
                t['band_options'] = _band(bo, base, where + '.band_options')
    else:
        if 'band_options' in t or 'measured_bandpass' in t:
            raise ValueError("%s: band_options and measured_bandpass are for presets; give the full Band "
                             "parameters in 'bands'" % where)
        if not isinstance(t.get('bands'), list) or not t['bands'] or not all(isinstance(b, dict) for b in t['bands']):
            raise ValueError("%s: give 'preset' (one of %s), or 'bands': a list of Band parameters, e.g. "
                             "{freq: 150, fwhm: 1.0, noise: 3, frac_bandwidth: 0.25}" % (where, ', '.join(sorted(PRESETS))))
        t['bands'] = [_band(b, base, '%s.bands[%d]' % (where, j)) for j, b in enumerate(t['bands'])]
    if isinstance(t.get('transfer'), str):
        t['transfer'] = _path(t['transfer'], base)
    for k in ('cmb', 'foregrounds'):
        if k in t:
            t[k] = _sky_path(t[k], base)
    return t


def _sky_path(v, base):
    """cmb ('camb' / file) and foregrounds (file / {band: file}) with resolved file names."""
    if isinstance(v, dict):
        return {k: _path(x, base) for k, x in v.items()}
    if isinstance(v, str) and v.lower() != 'camb':
        return _path(v, base)
    return v


def resolve_params(params, base_dir=None):
    """Check a parameter dict, fill in the defaults and make the file names absolute (relative to base_dir,
    default: the current directory). Returns a new dict."""
    base = os.path.abspath(base_dir or os.getcwd())
    p = _coerce(copy.deepcopy(dict(params)))
    _check_keys(p, tuple(DEFAULTS), 'the parameter file')
    out = {}
    for sec, dflt in DEFAULTS.items():
        if sec == 'telescopes':
            continue
        val = p.get(sec) or {}
        _check_keys(val, tuple(dflt), "section '%s'" % sec)
        out[sec] = copy.deepcopy(dflt)
        out[sec].update(val)
    tels = p.get('telescopes') or []
    if isinstance(tels, (str, dict)):
        tels = [tels]
    out['telescopes'] = [_telescope(t, base, i) for i, t in enumerate(tels)]

    sim = out['simulation']
    sim['snapshot'] = _path(sim['snapshot'], base)
    sim['catalogue'] = _path(sim['catalogue'], base)
    if sim['objects'] is not None:
        objs = sim['objects'] if isinstance(sim['objects'], list) else [sim['objects']]
        sim['objects'] = []
        for i, o in enumerate(objs):
            _check_keys(o, _OBJ_KEYS, 'simulation.objects[%d]' % i)
            o = dict(o)
            if o.get('snapshot') is not None:
                o['snapshot'] = _path(o['snapshot'], base)
            sim['objects'].append(o)
        if sim['catalogue'] is not None:
            raise ValueError("give simulation.objects or simulation.catalogue, not both")

    sz = out['sz']
    m = str(sz['model']).lower().replace(' ', '')
    if m not in _MODELS:
        raise ValueError("sz.model must be 'szpack', 'tt+tk', 'tt' or 'tk', not %r" % sz['model'])
    if m == 'tt+tk' and sz['kinetic'] is False:
        m = 'tt'
    if m == 'tk' and sz['kinetic'] is False:
        raise ValueError("sz.model 'tk' (kinetic SZ only) with sz.kinetic false leaves no signal")
    sz['model'] = _MODELS[m]
    if sz['model'] != 'szpack' and sz['bulk_velocity'] is not None:
        raise ValueError("sz.bulk_velocity is only used by sz.model 'szpack' (TK_model uses the velocities "
                         "relative to the halo)")

    obs = out['observation']
    obs['cmb'] = _sky_path(obs['cmb'], base)
    obs['foregrounds'] = _sky_path(obs['foregrounds'], base)
    if obs['seed'] is None:  # draw it now, so that the saved parameters reproduce the run
        obs['seed'] = int(np.random.SeedSequence().generate_state(1)[0])
    out['output']['directory'] = _path(out['output']['directory'], base)
    return out


# ------------------------------------------------------------------------------------------------
# objects, lines of sight, telescopes
# ------------------------------------------------------------------------------------------------
def _read_catalogue(fname):
    """Objects from a text catalogue whose first line names the columns: x y z radius (required), name,
    snapshot, redshift (optional). Other columns are ignored."""
    d = np.atleast_1d(np.genfromtxt(fname, names=True, dtype=None, encoding='utf-8', comments='#'))
    if d.dtype.names is None:
        raise ValueError("catalogue %s: the first line must name the columns, e.g. '# name x y z radius'" % fname)
    cols = {c.lower(): c for c in d.dtype.names}
    miss = [c for c in ('x', 'y', 'z', 'radius') if c not in cols]
    if miss:
        raise ValueError("catalogue %s has no column(s) %s (columns: %s)" % (fname, ', '.join(miss),
                                                                          ', '.join(d.dtype.names)))
    base = os.path.dirname(fname)
    objs = []
    for row in d:
        o = dict(center=[float(row[cols[c]]) for c in 'xyz'], radius=float(row[cols['radius']]))
        if 'name' in cols:
            o['name'] = str(row[cols['name']])
        if 'snapshot' in cols:
            o['snapshot'] = _path(str(row[cols['snapshot']]), base)
        if 'redshift' in cols:
            o['redshift'] = float(row[cols['redshift']])
        objs.append(o)
    return objs


def list_objects(params):
    """The objects to observe: dicts with name, snapshot, center, radius and optional image overrides."""
    sim = params['simulation']
    if sim['objects']:
        objs = [dict(o) for o in sim['objects']]
    elif sim['catalogue']:
        objs = _read_catalogue(sim['catalogue'])
    else:
        objs = [dict(name=sim['name'], center=sim['center'], radius=sim['radius'])]
    for i, o in enumerate(objs):
        if o.get('snapshot') is None:
            o['snapshot'] = sim['snapshot']
        if o['snapshot'] is None:
            raise ValueError("No snapshot for object %d: set simulation.snapshot" % i)
        if (o.get('center') is None) != (o.get('radius') is None):
            raise ValueError("Object %d: give both center and radius (kpc/h), or neither (whole snapshot)" % i)
        if o.get('name') is None:
            o['name'] = os.path.basename(o['snapshot'])
            for ext in ('.hdf5', '.HDF5', '.hdf', '.0'):
                o['name'] = o['name'][:-len(ext)] if o['name'].endswith(ext) else o['name']
        o['name'] = re.sub(r'[\s/\\]+', '_', str(o['name']))
    names = [o['name'] for o in objs]
    for n in set(names):
        if names.count(n) > 1:
            k = 0
            for o in objs:
                if o['name'] == n:
                    o['name'] = '%s_%d' % (n, k)
                    k += 1
    return objs


def projections(axis):
    r"""Lines of sight from the ``image.axis`` parameter.

    Parameters
    ----------
    axis : str, list or dict
        One of, or a list of:

        - 'x', 'y', 'z': along a simulation axis;
        - ``[alpha, beta, gamma]``: rotation by these angles (deg) around x, y, z (see
          :func:`pymsz.rotate_data.rotate_data`); a list of 3 numbers is always one rotation;
        - ``{angles: [...], label: name}``: the same, with a label for the file names;
        - ``{matrix: 3x3, label: name}``: a rotation matrix;
        - ``{random: N, seed: 0}``: N random orientations, uniform on the sphere (labels rand000,
          ...).

    Returns
    -------
    list of tuple
        ``(label, axis argument of the models)`` for every line of sight.
    """
    if axis is None:
        axis = 'z'
    items = axis if isinstance(axis, (list, tuple)) and not _is_angles(axis) else [axis]
    out = []
    for a in items:
        if isinstance(a, str):
            if a.lower() not in ('x', 'y', 'z'):
                raise ValueError("image.axis: unknown axis '%s' (x, y, z, angles or a dict)" % a)
            out.append((a.lower(), a.lower()))
        elif _is_angles(a):
            a = [float(v) for v in a]
            out.append(('a' + '_'.join('%g' % v for v in a), a))
        elif isinstance(a, dict):
            _check_keys(a, _AXIS_KEYS, 'image.axis')
            if 'random' in a:
                from scipy.spatial.transform import Rotation
                n = int(a['random'])
                q = np.random.default_rng(int(a.get('seed', 0))).standard_normal((n, 4))  # uniform rotations
                R = Rotation.from_quat(q / np.linalg.norm(q, axis=1)[:, None]).as_matrix().reshape(n, 3, 3)
                out += [('%s%03d' % (a.get('label', 'rand'), k), R[k]) for k in range(n)]
            elif 'angles' in a:
                if not _is_angles(a['angles']):
                    raise ValueError("image.axis: angles must be [alpha, beta, gamma] in degrees")
                ang = [float(v) for v in a['angles']]
                out.append((a.get('label', 'a' + '_'.join('%g' % v for v in ang)), ang))
            elif 'matrix' in a:
                m = np.asarray(a['matrix'], dtype=np.float64)
                if m.shape != (3, 3):
                    raise ValueError("image.axis: matrix must be 3x3")
                out.append((a.get('label', 'm%d' % len(out)), m))
            else:
                raise ValueError("image.axis: a dict needs 'angles', 'matrix' or 'random'")
        else:
            raise ValueError("image.axis: cannot understand %r" % (a,))
    labels = [o[0] for o in out]
    if len(set(labels)) != len(labels):
        raise ValueError("image.axis: the lines of sight have the same labels %s" % labels)
    return out


def _load_beam(beam):
    if isinstance(beam, str):
        d = np.loadtxt(beam)
        return d[:, 0], d[:, 1]
    return beam


def build_telescope(t):
    """(label, Telescope, observation overrides) from a resolved telescope entry."""
    t = dict(t)
    label = t.pop('label', None)
    over = {k: t.pop(k) for k in _TEL_OBS_KEYS if k in t}
    kw = {k: t.pop(k) for k in ('pixel', 'red_noise_corr', 'pointing_rms', 'description', 'reference') if k in t}
    if 'transfer' in t:
        tr = t.pop('transfer')
        if isinstance(tr, str):
            d = np.loadtxt(tr)
            tr = (d[:, 0], d[:, 1])
        kw['transfer'] = tr

    def band_args(d):
        d = dict(d)
        if 'beam' in d:
            d['beam'] = _load_beam(d['beam'])
        return d

    if 'preset' in t:
        bo = t.get('band_options')
        if bo:
            bo = {k: band_args(v) for k, v in bo.items()} if all(isinstance(v, dict) for v in bo.values()) \
                else band_args(bo)
        tel = Telescope.preset(t['preset'], bands=t.get('bands'), band_options=bo,
                               measured_bandpass=t.get('measured_bandpass', True), **kw)
        if t.get('name'):
            tel.name = t['name']
    else:
        tel = Telescope([Band(**band_args(b)) for b in t['bands']], name=t.get('name', 'custom'), **kw)
    return re.sub(r'[\s/\\]+', '_', str(label or tel.name)), tel, over


def _derive_seed(seed, *keys):
    """Reproducible seed for one map, from the run seed and names (independent of the loop order)."""
    ent = [int(seed)] + [zlib.crc32(str(k).encode()) for k in keys]
    return int(np.random.SeedSequence(ent).generate_state(1)[0])


def _zstr(z):
    return '%g' % z


def _plain(v):
    """numpy values -> Python values (for writing the parameters)."""
    if isinstance(v, dict):
        return {k: _plain(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_plain(x) for x in v]
    if isinstance(v, np.ndarray):
        return _plain(v.tolist())
    if isinstance(v, np.generic):
        return v.item()
    return v


# ------------------------------------------------------------------------------------------------
# the pipeline
# ------------------------------------------------------------------------------------------------
class Pipeline(object):
    r"""Mock SZ observations of simulated objects, configured by a parameter file or dict.

    :meth:`run` does the work; :meth:`describe` tells what it will do.

    Parameters
    ----------
    params : str or dict
        Parameter file name (.yaml, .toml, .json) or a dict with the sections of `DEFAULTS`.
    overrides : dict, optional
        ``{'section.key': value}`` applied before checking, e.g. ``{'image.npixel': 256}``.
    verbose : bool, optional
        Print the progress. Default True.

    Attributes
    ----------
    params : dict
        The checked parameters with all defaults (file names absolute).
    objects : list of dict
        The objects to observe (:func:`list_objects`).
    telescopes : list of tuple
        ``(label, Telescope, observation overrides)``.
    freqs : ndarray
        Frequencies (GHz) at which the SZ model is computed: the band-integration nodes of all
        telescopes plus ``sz.frequencies``.

    Examples
    --------
    >>> pipe = Pipeline('my_mock.yaml', overrides={'image.npixel': 256})
    >>> print(pipe.describe())
    >>> records = pipe.run()
    """

    def __init__(self, params, overrides=None, verbose=True):
        if isinstance(params, (str, os.PathLike)):
            self.source = os.path.abspath(str(params))
            raw = read_params(self.source)
            base = os.path.dirname(self.source)
        else:
            self.source = None
            raw = copy.deepcopy(dict(params))
            base = os.getcwd()
        for k, v in (overrides or {}).items():
            set_param(raw, k, v)
        self.verbose = verbose
        self.params = resolve_params(raw, base)
        self.telescopes = [build_telescope(t) for t in self.params['telescopes']]
        labels = [t[0] for t in self.telescopes]
        if len(set(labels)) != len(labels):
            raise ValueError("Telescope labels must be unique (%s): add 'label' to the telescope entries" % labels)
        self.objects = list_objects(self.params)
        for o in self.objects:
            projections(self._image(o)['axis'])  # check now, not after loading the first snapshot
        f = [tel.frequencies() for _, tel, _ in self.telescopes]
        if self.params['sz']['frequencies'] is not None:
            f.append(np.atleast_1d(np.asarray(self.params['sz']['frequencies'], dtype=np.float64)))
        if not f:
            raise ValueError("Nothing to compute: give telescopes (and/or sz.frequencies)")
        self.freqs = np.unique(np.round(np.concatenate(f), 9))
        self._cmb_cache = {}
        self._t0 = time.time()

    # --------------------------------------------------------------------------------------------
    def _log(self, msg):
        if self.verbose:
            print('[pymsz %7.1fs] %s' % (time.time() - self._t0, msg), flush=True)

    def _image(self, obj):
        img = dict(self.params['image'])
        img.update({k: obj[k] for k in DEFAULTS['image'] if k in obj})
        return img

    @staticmethod
    def _redshifts(img):
        z = img['redshift']
        if z is None:
            return [None]
        return [float(v) for v in z] if isinstance(z, (list, tuple)) else [float(z)]

    def model_description(self):
        sz = self.params['sz']
        if sz['model'] == 'szpack':
            return "SZpack %s (%s LOS), %s" % (sz['method'], sz['projection'], 'tSZ+kSZ' if sz['kinetic'] else 'tSZ')
        tt = 'TT(%s)' % ('rel. ' + sz['method'] if sz['relativistic'] else 'non-rel.')
        return {'tt+tk': tt + '+TK', 'tt': tt, 'tk': 'TK'}[sz['model']]

    def _model_names(self):
        return {'szpack': ['SZpack'], 'tt+tk': ['tSZ', 'kSZ'], 'tt': ['tSZ'], 'tk': ['kSZ']}[self.params['sz']['model']]

    def files(self, obj, z, proj):
        """Output files of one object, redshift and line of sight: ({telescope: file}, {model: file})."""
        out = self.params['output']
        f = dict(object=obj['name'], redshift=_zstr(z), projection=proj)

        def fn(fmt, **kw):
            name = fmt.format(**dict(f, **kw))
            return os.path.join(out['directory'], name if name.endswith('.fits') else name + '.fits')
        tel = {lab: fn(out['name_format'], telescope=lab) for lab, _, _ in self.telescopes}
        mod = {m: fn(out['model_format'], model=m) for m in self._model_names()} if out['save_model'] else {}
        return tel, mod

    # --------------------------------------------------------------------------------------------
    def describe(self):
        """Text summary of what run() will do."""
        P = self.params
        L = ["pymsz mock observation" + (" from %s" % self.source if self.source else '')]
        L.append("Objects (%d):" % len(self.objects))
        for o in self.objects[:20]:
            snap = o['snapshot']
            found = os.path.exists(snap) or any(os.path.exists(snap + e) for e in ('.0', '.0.hdf5', '.hdf5'))
            s = "  %-16s %s%s" % (o['name'], snap, '' if found else '  (NOT FOUND)')
            if o.get('center') is not None:
                s += "\n  %16s center [%s] kpc/h, radius %g kpc/h" % (
                    '', ', '.join('%.10g' % float(v) for v in o['center']), float(o['radius']))
            over = {k: o[k] for k in DEFAULTS['image'] if k in o}
            if over:
                s += "\n  %16s %s" % ('', over)
            L.append(s)
        if len(self.objects) > 20:
            L.append("  ... and %d more" % (len(self.objects) - 20))
        img = P['image']
        zs = self._redshifts(img)
        L.append("Images: redshift %s, %s pixels of %s arcsec, SPH kernel %s" % (
            'of the snapshot' if zs == [None] else ', '.join(_zstr(z) for z in zs), img['npixel'],
            img['AR'] if img['AR'] is not None else '(2 radius / npixel)', img['sph_kernel']))
        L.append("Lines of sight: %s" % ', '.join(p[0] for p in projections(img['axis'])))
        L.append("SZ model: %s, at %d frequencies (%.4g-%.4g GHz)" % (self.model_description(), self.freqs.size,
                                                                      self.freqs[0], self.freqs[-1]))
        o = P['observation']
        L.append("Observation: unit %s, noise %s, CMB %s, calibration errors %s, seed %s" % (
            o['unit'], 'on' if o['noise'] else 'off', o['cmb'], 'on' if o['calibration'] else 'off', o['seed']))
        L.append("Telescopes (%d):" % len(self.telescopes))
        for lab, tel, over in self.telescopes:
            L.append("  [%s] " % lab + repr(tel).replace('\n', '\n    ') +
                     ("\n    overrides: %s" % over if over else ''))
        nmap = sum(len(self._redshifts(self._image(ob))) * len(projections(self._image(ob)['axis']))
                   for ob in self.objects)
        L.append("Output: %s" % P['output']['directory'])
        if self.objects:
            ob = self.objects[0]
            z0 = self._redshifts(self._image(ob))[0]
            tf, mf = self.files(ob, 0.1 if z0 is None else z0, projections(self._image(ob)['axis'])[0][0])
            for f in list(tf.values())[:2] + list(mf.values())[:2]:
                L.append("  e.g. %s" % os.path.basename(f))
        L.append("Total: %d SZ models, %d mock observations" % (nmap, nmap * len(self.telescopes)))
        return '\n'.join(L)

    # --------------------------------------------------------------------------------------------
    def write_params(self, fname=None):
        """Write the checked parameters (with the defaults and the drawn seed) to the output directory."""
        out = self.params['output']
        stem = os.path.splitext(os.path.basename(self.source))[0] if self.source else 'pymsz_mock'
        try:
            yaml = _yaml()
        except ImportError:
            yaml = None
        if fname is None:
            fname = os.path.join(out['directory'], stem + ('.resolved.yaml' if yaml else '.resolved.json'))
        from pymsz import version
        if yaml is not None:
            with open(fname, 'w') as f:
                f.write("# pymsz %s mock observation parameters, written %s\n# from %s\n" % (
                    version.version, time.strftime('%Y-%m-%d %H:%M:%S'), self.source or 'a dict'))
                yaml.safe_dump(_plain(self.params), f, sort_keys=False, default_flow_style=None)
        else:
            with open(fname, 'w') as f:
                json.dump(_plain(self.params), f, indent=2)
        return fname

    def load(self, obj):
        """load_data for one object."""
        from pymsz.load_data import load_data
        sim = self.params['simulation']
        cut = sim['cut_rhoT'] if sim['cut_rhoT'] is not None else [None, None]
        center = None if obj.get('center') is None else np.asarray(obj['center'], dtype=np.float64)
        return load_data(filename=obj['snapshot'], snapshot=True, center=center, radius=obj.get('radius'),
                         metal=sim['metal'], Nmets=sim['Nmets'], mu=sim['mu'], restrict_r=sim['restrict_r'],
                         hmrad=sim['hmrad'], cut_sfr=sim['cut_sfr'], cut_rhoT=cut)

    def model(self, simd, img, z, axis):
        """SZ model(s) of one object at redshift z along axis: {name: model}, name in SZpack, tSZ, kSZ."""
        from pymsz.Theoretical_models import TT_model, TK_model
        from pymsz.SZpack_models import SZpack_model
        sz = self.params['sz']
        geo = dict(npixel=img['npixel'], neighbours=img['neighbours'], axis=axis, AR=img['AR'], SD=img['SD'],
                   SP=img['SP'], Memreduce=img['Memreduce'], Ncpu=img['Ncpu'], Ntasks=img['Ntasks'], redshift=z,
                   zthick=img['zthick'], sph_kernel=img['sph_kernel'])
        if sz['model'] == 'szpack':
            bv = sz['bulk_velocity']
            if bv is not None and not isinstance(bv, bool):
                bv = np.asarray(bv, dtype=np.float64)
            mm = SZpack_model(simd, self.freqs, method=sz['method'], projection=sz['projection'],
                              kinetic=bool(sz['kinetic']), bulk_velocity=bv, unit=sz['unit'],
                              szpack_options=sz['szpack_options'], Te_grid=sz['Te_grid'], **geo)
            return {'SZpack': mm}
        comps = {}
        if sz['model'] in ('tt+tk', 'tt'):
            tt = TT_model(simd, yweighted_Te=bool(sz['relativistic']), **geo)
            tt.to_sz(self.freqs, unit=sz['unit'], relativistic=bool(sz['relativistic']), method=sz['method'],
                     szpack_options=sz['szpack_options'])
            comps['tSZ'] = tt
        if sz['model'] in ('tt+tk', 'tk'):
            tk = TK_model(simd, **geo)
            tk.to_sz(self.freqs, unit=sz['unit'])
            comps['kSZ'] = tk
        return comps

    def _cmb(self, cmb):
        if cmb is None or cmb is False:
            return None
        key = repr(cmb)
        if key not in self._cmb_cache:
            self._log("CMB power spectrum: %s" % (cmb,))
            self._cmb_cache[key] = cmb_power_spectrum(cmb)
        return self._cmb_cache[key]

    @staticmethod
    def _foregrounds(fg, tel):
        if fg is None:
            return None
        import astropy.io.fits as pf
        if isinstance(fg, dict):
            names = [b.name for b in tel.bands]
            for k in fg:
                if k not in names:
                    raise ValueError("foregrounds for band '%s', but the bands of %s are %s" % (k, tel.name, names))
            return {k: np.asarray(pf.getdata(v), dtype=np.float64) for k, v in fg.items()}
        return np.asarray(pf.getdata(fg), dtype=np.float64)

    # --------------------------------------------------------------------------------------------
    def run(self, keep_results=False, callback=None):
        r"""Run all objects x redshifts x lines of sight x telescopes.

        Parameters
        ----------
        keep_results : bool, optional
            Also return the `MockObservation` objects (key 'observation' of the records). Default
            False: they can be large for big runs, and the maps are in the files.
        callback : callable, optional
            Function ``f(obs, info)`` called after every observation, with `info` the record (object,
            redshift, projection, telescope, file, ...) plus 'models' (the SZ models).

        Returns
        -------
        list of dict
            One record per mock observation, with the keys object, redshift, projection, telescope,
            file, model_files, seed, cmb_seed and observation.
        """
        P = self.params
        out = P['output']
        os.makedirs(out['directory'], exist_ok=True)
        if out['save_params']:
            self._log("parameters written to %s" % self.write_params())
        self._log("%d object(s); SZ model %s at %d frequencies; telescopes: %s" % (
            len(self.objects), self.model_description(), self.freqs.size, ', '.join(t[0] for t in self.telescopes)))
        records = []
        for obj in self.objects:
            img = self._image(obj)
            projs = projections(img['axis'])
            simd = None
            for z in self._redshifts(img):
                if simd is None and (z is None or not self._done(obj, z, projs)):
                    self._log("loading %s from %s" % (obj['name'], obj['snapshot']))
                    simd = self.load(obj)
                    self._log("loaded %d gas particles" % np.size(simd.temp))
                zz = float(simd.cosmology['z']) if z is None else z
                if zz <= 0:
                    raise ValueError("The object must be placed at redshift > 0 for a mock observation (image.redshift;"
                                     " the snapshot is at z = %g)" % zz)
                for plab, pax in projs:
                    records += self._one(simd, obj, img, zz, plab, pax, keep_results, callback)
            del simd
        self._log("done: %d mock observations in %s" % (len(records), out['directory']))
        return records

    def _done(self, obj, z, projs):
        if not self.params['output']['skip_existing']:
            return False
        for plab, _ in projs:
            tf, mf = self.files(obj, z, plab)
            if not all(os.path.exists(f) for f in list(tf.values()) + list(mf.values())):
                return False
        return True

    def _one(self, simd, obj, img, z, plab, pax, keep_results, callback):
        P = self.params
        out = P['output']
        tf, mf = self.files(obj, z, plab)
        tag = "%s z=%s %s" % (obj['name'], _zstr(z), plab)
        exist = [f for f in list(tf.values()) + list(mf.values()) if os.path.exists(f)]
        if exist and not (out['overwrite'] or out['skip_existing']):
            raise FileExistsError("%s exists. Set output.overwrite or output.skip_existing to true, or change "
                                  "output.directory" % exist[0])
        todo = [t for t in self.telescopes if not (out['skip_existing'] and os.path.exists(tf[t[0]]))]
        mtodo = [m for m in mf if not (out['skip_existing'] and os.path.exists(mf[m]))]
        if not todo and not mtodo:
            self._log("%s: all files exist, skipped" % tag)
            return []

        self._log("%s: SZ model %s" % (tag, self.model_description()))
        models = self.model(simd, img, z, pax)
        sky = list(models.values())
        ar = float(sky[0].ar)
        self._log("%s: %d x %d pixels of %.4g arcsec" % (tag, sky[0].szdata.shape[-2], sky[0].szdata.shape[-1], ar))
        com = ['object %s, snapshot %s' % (obj['name'], os.path.basename(obj['snapshot'])),
               'line of sight %s, redshift %s' % (plab, _zstr(z)),
               'pymsz parameter file %s' % (os.path.basename(self.source) if self.source else 'dict')]
        for m in mtodo:
            if m == 'SZpack':
                models[m].write_fits_image(mf[m], comments=com, overwrite=True)
            else:
                models[m].write_sz_fits(mf[m], comments=com, overwrite=True)
            self._log("%s: wrote %s" % (tag, mf[m]))

        recs = []
        seed = P['observation']['seed']
        cmb_seed = _derive_seed(seed, obj['name'], _zstr(z), plab, 'CMB')  # same CMB sky for all telescopes
        for lab, tel, over in todo:
            o = dict(P['observation'])
            o.update(over)
            s = _derive_seed(seed, obj['name'], _zstr(z), plab, lab)
            obs = tel.observe(sky, unit=o['unit'], noise=bool(o['noise']), cmb=self._cmb(o['cmb']),
                              foregrounds=self._foregrounds(o['foregrounds'], tel), calibration=bool(o['calibration']),
                              seed=s, pad=o['pad'], cmb_seed=cmb_seed)
            hdr = {'OBJECT': (obj['name'][:40], 'simulated object'),
                   'PROJ': (plab[:40], 'line of sight'),
                   'SZMODEL': (self.model_description()[:40], 'SZ model of the sky'),
                   'SNAPSHOT': (os.path.basename(obj['snapshot'])[-40:], 'simulation snapshot'),
                   'PARFILE': ((os.path.basename(self.source) if self.source else 'dict')[-40:], 'pymsz parameters')}
            obs.write_fits(tf[lab], components=bool(out['components']), overwrite=True, header=hdr)
            self._log("%s: %s -> %s" % (tag, lab, tf[lab]))
            rec = dict(object=obj['name'], redshift=z, projection=plab, telescope=lab, file=tf[lab],
                       model_files=dict(mf), seed=s, cmb_seed=cmb_seed if obs.cmb is not None else None,
                       observation=obs if keep_results else None)
            if callback is not None:
                callback(obs, dict(rec, observation=obs, models=models))
            recs.append(rec)
        return recs


def run(params, overrides=None, keep_results=False, callback=None, verbose=True):
    r"""Run the mock observations of a parameter file (or dict).

    The same as ``Pipeline(params, overrides, verbose).run(keep_results, callback)``; see
    `Pipeline` and :meth:`Pipeline.run`.

    Returns
    -------
    list of dict
        One record per mock observation.

    Examples
    --------
    >>> recs = pymsz.pipeline.run('my_mock.yaml', overrides={'image.npixel': 256})
    """
    return Pipeline(params, overrides=overrides, verbose=verbose).run(keep_results=keep_results, callback=callback)


def write_template(fname, overwrite=False):
    """Copy the annotated parameter template to fname."""
    if os.path.exists(fname) and not overwrite:
        raise FileExistsError("%s exists" % fname)
    shutil.copyfile(TEMPLATE, fname)
    return fname


# ------------------------------------------------------------------------------------------------
# command line
# ------------------------------------------------------------------------------------------------
def main(argv=None):
    """The command ``pymsz-mock`` (and ``python -m pymsz``). Returns the exit status."""
    ap = argparse.ArgumentParser(
        prog='pymsz-mock',
        description="Mock SZ observations of simulated objects (snapshot -> SZ model -> telescope maps) "
                    "configured by a parameter file (.yaml, .toml or .json).",
        epilog="Start with: pymsz-mock --template my_mock.yaml; edit it; pymsz-mock my_mock.yaml --dry-run")
    ap.add_argument('params', nargs='?', help='parameter file')
    ap.add_argument('--template', metavar='FILE', help='write an annotated template parameter file and exit')
    ap.add_argument('-n', '--dry-run', action='store_true',
                    help='check the parameters and print what would be done, without computing')
    ap.add_argument('-s', '--set', dest='overrides', action='append', default=[], metavar='KEY=VALUE',
                    help="override a parameter, e.g. --set image.npixel=256 --set 'telescopes=[SPT-3G]' "
                         "(repeatable; values in YAML syntax)")
    ap.add_argument('-q', '--quiet', action='store_true', help='do not print the progress')
    ap.add_argument('--list', choices=['presets', 'bandpasses'], help='list the telescope presets or the '
                    'shipped bandpasses and exit')
    args = ap.parse_args(argv)

    if args.list == 'presets':
        for k in sorted(PRESETS):
            print("%-12s %s" % (k, PRESETS[k].get('description', '')))
        return 0
    if args.list == 'bandpasses':
        print('\n'.join(bandpasses.available()))
        return 0
    if args.template:
        try:
            write_template(args.template)
        except FileExistsError as e:
            ap.error(str(e))
        print("wrote %s: edit it, then run  pymsz-mock %s --dry-run" % (args.template, args.template))
        return 0
    if not args.params:
        ap.error("a parameter file is needed (start with --template FILE)")
    ov = {}
    for s in args.overrides:
        if '=' not in s:
            ap.error("--set needs KEY=VALUE, got %r" % s)
        k, v = s.split('=', 1)
        ov[k.strip()] = parse_value(v)
    try:
        pipe = Pipeline(args.params, overrides=ov, verbose=not args.quiet)
    except (ValueError, KeyError, OSError, ImportError) as e:
        print("pymsz-mock: error: %s" % e, file=sys.stderr)
        return 2
    if args.dry_run:
        print(pipe.describe())
        return 0
    pipe.run()
    return 0


if __name__ == '__main__':
    sys.exit(main())
