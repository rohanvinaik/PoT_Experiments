# Attack Utilities

`pot_attack.sh` is a thin wrapper around the `pot.cli.attack_cli` module.
It sets up the Python path and provides easy command-line access to the
attack evaluation tools in this repository.

## Usage

Run the script from the repository root:

```bash
bash scripts/attack/pot_attack.sh run-attacks -m model.pth -s standard
bash scripts/attack/pot_attack.sh benchmark -c config.yaml -m model.pth
```

For additional commands and options:

```bash
bash scripts/attack/pot_attack.sh --help
```

