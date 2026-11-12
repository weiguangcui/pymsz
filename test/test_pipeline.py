"""Parameter-file interface (pymsz.pipeline, pymsz-mock): checking of the parameters, the command
line, and full runs snapshot -> SZ model -> mock telescope maps on the test cluster."""
import json
import os

import numpy as np
import pytest

from conftest import CL288, GEO, HERE, NCPU, SNAPSHOT
from pymsz import pipeline
from pymsz.pipeline import Pipeline, main, parse_value, projections, set_param

EXAMPLE = os.path.join(HERE, 'mock_CL288.yaml')


def base_params(tmp_path, **sections):
    """A small run on the test cluster; sections override whole parameter sections."""
    p = {'simulation': dict(snapshot=SNAPSHOT, name='CL288', **CL288),
         'image': dict(GEO),
         'sz': {'model': 'tt+tk', 'relativistic': False},
         'telescopes': ['SPT-3G'],
         'observation': {'seed': 3},
         'output': {'directory': str(tmp_path / 'out')}}
    p.update(sections)
    return p


# ------------------------------------------------------------------------------------------------
# parameters (no simulation data needed)
# ------------------------------------------------------------------------------------------------
def test_template_and_example_are_valid(tmp_path, capsys):
    t = str(tmp_path / 'template.yaml')
    assert main(['--template', t]) == 0
    with pytest.raises(FileExistsError):
        pipeline.write_template(t)  # no overwrite
    assert main([t, '--dry-run']) == 0
    out = capsys.readouterr().out
    assert 'NOT FOUND' in out and 'SPT-3G' in out and 'ACT-DR6' in out  # the template snapshot is a placeholder
    assert main([EXAMPLE, '--dry-run', '--set', 'image.npixel=64']) == 0
    assert 'NOT FOUND' not in capsys.readouterr().out


@pytest.mark.parametrize('key,value,match', [
    ('image.npixle', 64, "Did you mean 'npixel'"),
    ('sz.model', 'szpak', 'szpack'),
    ('image.axis', 'w', 'unknown axis'),
    ('telescopes', ['SPT-4G'], 'SPT-3G'),
    ('telescopes', [{'preset': 'ACT-DR6', 'band_options': {'f100': {'shift': 1.}}}], 'f090'),
    ('telescopes', [{'name': 'Cam', 'bands': [{'freq': 150., 'fwhm': 1., 'bandpass': 'SPT-3G/f15'}]}], 'SPT-3G/f150'),
    ('telescopes', [{'name': 'Cam', 'bands': [{'freq': 150., 'fhwm': 1.}]}], "fwhm"),
    ('telescopes', [], 'telescope'),
])
def test_parameter_errors(tmp_path, key, value, match):
    with pytest.raises(ValueError, match=match):
        Pipeline(base_params(tmp_path), overrides={key: value}, verbose=False)


def test_command_line_errors(tmp_path, capsys):
    f = tmp_path / 'bad.json'
    f.write_text(json.dumps(base_params(tmp_path, image={'npixle': 64})))
    assert main([str(f), '--dry-run']) == 2
    assert "pymsz-mock: error" in capsys.readouterr().err
    assert main(['--list', 'presets']) == 0
    assert 'SPT-3G' in capsys.readouterr().out
    assert main(['--list', 'bandpasses']) == 0
    assert 'Planck/HFI143' in capsys.readouterr().out


def test_parse_and_set():
    assert parse_value('1e-4') == 1e-4 and isinstance(parse_value('1e-4'), float)
    assert parse_value('[SPT-3G, Planck]') == ['SPT-3G', 'Planck']
    assert parse_value('null') is None and parse_value('z') == 'z'
    p = {'telescopes': ['SPT-3G'], 'image': {}}
    set_param(p, 'telescopes.0.pixel', None)
    set_param(p, 'image.npixel', 64)
    assert p == {'telescopes': [{'preset': 'SPT-3G', 'pixel': None}], 'image': {'npixel': 64}}


def test_projections():
    assert projections('x') == [('x', 'x')]
    (lab, ang), = projections([0., 45., 30.])
    assert lab == 'a0_45_30' and ang == [0., 45., 30.]
    rand = projections([{'random': 3, 'seed': 1}, 'z'])
    assert [r[0] for r in rand] == ['rand000', 'rand001', 'rand002', 'z']
    for _, m in rand[:3]:
        assert np.allclose(m @ m.T, np.eye(3)) and np.linalg.det(m) == pytest.approx(1.)
    assert np.allclose(projections({'random': 3, 'seed': 1})[1][1], rand[1][1])  # reproducible
    with pytest.raises(ValueError):
        projections(['x', 'x'])


def test_formats_are_equivalent(tmp_path):
    p = base_params(tmp_path)
    (tmp_path / 'p.json').write_text(json.dumps(p))
    yaml = pytest.importorskip('yaml')
    (tmp_path / 'p.yaml').write_text(yaml.safe_dump(p))
    a = Pipeline(str(tmp_path / 'p.json'), verbose=False).params
    b = Pipeline(str(tmp_path / 'p.yaml'), verbose=False).params
    cen = CL288['center']
    toml = """telescopes = ["SPT-3G"]
[simulation]
snapshot = "%s"
name = "CL288"
center = [%r, %r, %r]
radius = %r
restrict_r = false
cut_sfr = 0.1
[image]
npixel = 48
AR = 40.0
redshift = 0.1
axis = "z"
Ncpu = %d
[sz]
model = "tt+tk"
relativistic = false
[observation]
seed = 3
[output]
directory = "%s"
""" % (SNAPSHOT, cen[0], cen[1], cen[2], CL288['radius'], NCPU, tmp_path / 'out')
    (tmp_path / 'p.toml').write_text(toml)
    c = Pipeline(str(tmp_path / 'p.toml'), verbose=False).params
    assert a == b == c


def test_catalogue(tmp_path):
    (tmp_path / 'cat.txt').write_text('# name snapshot x y z radius redshift\n'
                                      'A snaps/snap_1 1. 2. 3. 100. 0.2\n'
                                      'B snaps/snap_2 4. 5. 6. 200. 0.3\n')
    p = base_params(tmp_path)
    p['simulation'] = {'catalogue': str(tmp_path / 'cat.txt')}
    pipe = Pipeline(p, verbose=False)
    assert [o['name'] for o in pipe.objects] == ['A', 'B']
    assert pipe.objects[1]['snapshot'] == str(tmp_path / 'snaps' / 'snap_2')
    assert np.allclose(pipe.objects[1]['center'], [4., 5., 6.]) and pipe.objects[1]['radius'] == 200.
    assert pipe.objects[0]['redshift'] == pytest.approx(0.2)


def test_seeds():
    s = pipeline._derive_seed(1, 'CL288', '0.1', 'z', 'SPT-3G')
    assert s == pipeline._derive_seed(1, 'CL288', '0.1', 'z', 'SPT-3G')
    assert s != pipeline._derive_seed(1, 'CL288', '0.1', 'z', 'ACT-DR6')
    assert s != pipeline._derive_seed(2, 'CL288', '0.1', 'z', 'SPT-3G')


# ------------------------------------------------------------------------------------------------
# full runs on the test cluster
# ------------------------------------------------------------------------------------------------
def test_run_matches_the_direct_api(tmp_path, tt, tk):
    """snapshot -> TT+TK -> SPT-3G and ACT-DR6 f150 with CMB and noise, against the same steps by hand."""
    from astropy.io import fits
    from pymsz import Telescope
    if not os.path.exists(SNAPSHOT):
        pytest.skip('test snapshot not found')
    ell = np.arange(2, 12001.)
    np.savetxt(tmp_path / 'cmb.txt', np.c_[ell, np.full_like(ell, 1000.)])  # CAMB-style: ell, D_ell (uK^2)
    p = base_params(tmp_path, telescopes=['SPT-3G', {'preset': 'ACT-DR6', 'bands': ['f150'], 'label': 'ACT150'}])
    p['observation'] = {'seed': 3, 'cmb': str(tmp_path / 'cmb.txt')}
    pipe = Pipeline(p, verbose=False)
    recs = pipe.run(keep_results=True)
    assert [r['telescope'] for r in recs] == ['SPT-3G', 'ACT150']
    assert recs[0]['cmb_seed'] == recs[1]['cmb_seed'] and recs[0]['seed'] != recs[1]['seed']

    # the same by hand
    out = tmp_path / 'out'
    tt.to_sz(pipe.freqs, unit='MJy/sr')
    tk.to_sz(pipe.freqs, unit='MJy/sr')
    for rec, tel in zip(recs, [Telescope.preset('SPT-3G'), Telescope.preset('ACT-DR6', bands=['f150'])]):
        obs = tel.observe([tt, tk], cmb=str(tmp_path / 'cmb.txt'), seed=rec['seed'], cmb_seed=rec['cmb_seed'])
        assert np.allclose(obs.maps, rec['observation'].maps, rtol=1e-9, atol=1e-9 * np.abs(obs.maps).max())
        assert rec['file'] == str(out / ('CL288_z0.1_z_%s.fits' % rec['telescope']))
        with fits.open(rec['file']) as h:
            assert [x.name for x in h] == ['PRIMARY', 'SZ', 'CMB', 'NOISE', 'SZTRUE']
            hdr = h[0].header
            assert (hdr['OBJECT'], hdr['PROJ'], int(hdr['SEED'])) == ('CL288', 'z', rec['seed'])
            assert int(hdr['CMBSEED']) == rec['cmb_seed']
    for m in ('tSZ', 'kSZ'):
        assert (out / ('CL288_z0.1_z_%s.fits' % m)).exists()

    # the saved parameters reproduce the run; existing files are refused, or skipped
    saved = [f for f in os.listdir(out) if '.resolved.' in f]
    assert len(saved) == 1
    assert Pipeline(str(out / saved[0]), verbose=False).params == pipe.params
    with pytest.raises(FileExistsError):
        Pipeline(p, verbose=False).run()
    assert Pipeline(p, overrides={'output.skip_existing': True}, verbose=False).run() == []


def test_szpack_run_from_the_example_file(tmp_path, szpack):
    """The shipped example test/mock_CL288.yaml, made small with --set, through the command line."""
    from astropy.io import fits
    if not os.path.exists(SNAPSHOT):
        pytest.skip('test snapshot not found')
    out = tmp_path / 'mock'
    assert main([EXAMPLE, '-q', '--set', 'image.npixel=48', '--set', 'image.AR=40', '--set', 'image.Ncpu=%d' % NCPU,
                 '--set', 'telescopes=[{name: Mono, bands: [{freq: 150., fwhm: 1.2, noise: 5.}]}]',
                 '--set', 'output.directory=%s' % out]) == 0
    with fits.open(out / 'CL288_z0.1_z_SZpack.fits') as h:
        assert h[0].header['FREQ1'] == 150.
    with fits.open(out / 'CL288_z0.1_z_Mono.fits') as h:
        sz = h['SZ'].data[0]
        noise = h['NOISE'].data[0]
    assert sz.min() < -100.  # uK_CMB, tSZ decrement at 150 GHz
    assert np.std(noise) == pytest.approx(5. / (40. / 60.), rel=0.1)
