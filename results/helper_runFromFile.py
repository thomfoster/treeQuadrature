import argparse
import subprocess
import yaml

parser = argparse.ArgumentParser(
    description="Run a list of experiments in a yaml config file.",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)
parser.add_argument('file', type=str, help="the config file to be used")
args = parser.parse_args()

with open(args.file, 'r') as f:
    d = yaml.safe_load(f)

defaults = d.get("defaults", {})
verbose_default = defaults.pop('verbose', False)
remove_nonce_default = defaults.pop('remove_nonce', False)

args_for_runOnMultipleCores = [
    'target',
    'args_string',
    'verbose',
    'num_runs',
    'num_cores',
    'key',
    'remove_nonce'
]

for experiment in d['experiments']:
    # prepare system call to helper_runOnMultipleCores.py
    args1 = []  # for runOnMultipleCores
    args2 = []  # for the test_XXX.py that runOnMultipleCores will run

    verbose = experiment.pop('verbose', verbose_default)
    if verbose:
        args1.append("--verbose")

    remove_nonce = experiment.pop('remove_nonce', remove_nonce_default)
    if remove_nonce:
        args1.append("--remove_nonce")

    for k,v in defaults.items():
        if k in args_for_runOnMultipleCores:
            args1.append("--"+str(k))
            args1.append(str(v))
        else:
            args2.append("--"+str(k))
            args2.append(str(v))

    for k,v in experiment.items():
        if k in args_for_runOnMultipleCores:
            args1.append("--" + str(k))
            args1.append(str(v))
        else:
            args2.append("--" + str(k))
            args2.append(str(v))

    system_call = ["python", "helper_runOnMultipleCores.py"] + args1 + ["--args_string", " ".join(args2)]
    subprocess.run(system_call)