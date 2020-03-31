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

args_for_runOnMultipleCores = [
    'target',
    'args_string',
    'verbose',
    'num_runs',
    'num_cores',
    'key',
    'add_nonce'
]

def format_dict_as_args(d):
    args = []
    for k, v in d.items():
        if k == "verbose":
            if v == "True":
                args.append("--" + str(k))
        else:
            args.append("--" + str(k))
            args.append(str(v))
    return args

for experiment in d['experiments']:
    # prepare system call to helper_runOnMultipleCores.py
    args1 = format_dict_as_args(defaults)  # for runOnMultipleCores
    args2 = []  # for the test_XXX.py that runOnMultipleCores will run
    for k,v in experiment.items():
        if k in args_for_runOnMultipleCores:
            args1.append("--" + str(k))
            args1.append(str(v))
        else:
            args2.append("--" + str(k))
            args2.append(str(v))
    system_call = ["python", "helper_runOnMultipleCores.py"] + args1 + ["--args_string", " ".join(args2)]
    print(system_call)
    subprocess.run(system_call)