# Usage: python3 runOnMultipleCores.py <script_name> <number of times> <key>

# This script runs the target script T times.
# The target script must accept a command line arg of key, which is
# used to group similar runs.

import subprocess
import argparse
import uuid
import sys

from multiprocessing import Pool
from tqdm import tqdm

parser = argparse.ArgumentParser(
    description='Run target script on multiple cores.',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)
parser.add_argument('--target', type=str, help="the script to be run by subprocess")
parser.add_argument('--args_string', type=str, default="", help="the string of args to be passed to the subprocess")
parser.add_argument('--verbose', action="store_true", help="Display the interleaved output of the subprocesses.")
parser.add_argument('--num_runs', type=int, default=20, help="the total number of times to run the target script")
parser.add_argument('--num_cores', type=int, default=6, help="size of worker pool")
parser.add_argument('--key', type=str, default="", help="A common string to be passed to each process. Used to group runs together.")
parser.add_argument('--add_nonce', type=bool, default=True, help='append a random unique string to the "key" variable, to ensure that this batch of runs all have unique keys.')
args = parser.parse_args()

print()
print('Running with args: ')
for k,v in vars(args).items():
    print(k, v)
print()

# Post processing of key parameter
assert (args.key != "") or (args.add_nonce), "Either provide a key using '--key my_key' or disable --add_nonce False to ensure key uniqueness."
if args.add_nonce:
    args.key += ' ' + str(uuid.uuid4()).split('-')[0]

# The function that runs on each core -
# performs system command line call to run the target script
def f(run_number):
    if len(args.args_string) > 0:
        result = subprocess.run(['python3', args.target, '--key', args.key, *args.args_string.split(' ')], capture_output=True)
    else:
        result = subprocess.run(['python3', args.target, '--key', args.key], capture_output=True)

    if args.verbose:
        print(f"\n\nOutput of {run_number}: \n\n")
        print(result.stderr.decode('utf-8'))
    else:
        print(f"Run {run_number} complete.")
    

# Start multiple processes on multiple cores
with Pool(processes=args.num_cores) as pool:
    run_numbers = [i for i in range(args.num_runs)]
    pool.map(f, run_numbers)