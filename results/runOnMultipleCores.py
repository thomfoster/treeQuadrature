# Usage: python3 runOnMultipleCores.py <script_name> <number of times> <key>

# This script runs the target script T times.
# The target script must accept a command line arg of key, which is
# used to group similar runs.

import subprocess
import argparse
import uuid

from multiprocessing import Pool
from tqdm import tqdm

parser = argparse.ArgumentParser(
    description='Run target script on multiple cores.',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)
parser.add_argument('target', type=str, help="relative path to the script to be run")
parser.add_argument('--num_runs', type=int, default=20, help="the total number of times to run the target script")
parser.add_argument('--num_cores', type=int, default=6, help="size of worker pool")
parser.add_argument('--key', type=str, default="", help="A common string to be passed to each process. Used to group runs together.")
parser.add_argument('--add_nonce', type=bool, default=True, help='append a random unique string to the "key" variable, to ensure that this batch of runs all have unique keys.')
args = parser.parse_args()

# Post processing of key parameter
assert (args.key != "") or (args.add_nonce), "Either provide a key using '--key my_key' or disable --add_nonce False to ensure key uniqueness."
if args.add_nonce:
    args.key += '-' + str(uuid.uuid4()).split('-')[0]

# The function that runs on each core -
# performs system command line call to run the target script
def f(run_number):
    subprocess.run(['python3', args.target, args.key])

# Start multiple processes on multiple cores
with Pool(processes=args.num_cores) as pool:
    run_numbers = [i for i in range(args.num_runs)]
    pool.map(f, run_numbers)