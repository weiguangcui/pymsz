# Command line: `pymsz-mock`

```text
pymsz-mock [-h] [--template FILE] [-n] [-s KEY=VALUE] [-q] [--list {presets,bandpasses}] [params]
```

`python -m pymsz` is the same command.

| Option | Meaning |
|---|---|
| `params` | the parameter file (`.yaml`/`.yml`, `.toml` or `.json`) |
| `--template FILE` | write the annotated template parameter file and exit |
| `-n`, `--dry-run` | check the parameters and print what would be done, without computing |
| `-s KEY=VALUE`, `--set KEY=VALUE` | override a parameter; repeatable; values in YAML syntax |
| `-q`, `--quiet` | do not print the progress |
| `--list presets` | list the telescope presets and exit |
| `--list bandpasses` | list the shipped bandpasses and exit |
| `-h`, `--help` | show the help |

## Overriding parameters

`--set` takes a dotted key. List items are addressed by their index:

```bash
pymsz-mock my_mock.yaml --set image.npixel=256 --set observation.seed=7
pymsz-mock my_mock.yaml --set 'image.redshift=[0.1, 0.2]'
pymsz-mock my_mock.yaml --set 'telescopes=[SPT-3G, Planck]'
pymsz-mock my_mock.yaml --set telescopes.0.pixel=0.5      # first telescope
```

Values are read as YAML if PyYAML is installed, else as JSON, else as a string.

## Exit status

`0` on success. `2` if the parameters are invalid: the error is printed to standard error, e.g.

```text
pymsz-mock: error: Unknown parameter 'npixels' in section 'image'. Did you mean 'npixel'? Allowed: ...
```
