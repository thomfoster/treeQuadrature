# Usage: python3 runSciptMultipleTimes.py <script_name> <number of times>

# This script runs the target script T times.
# The target script must accept a command line arg of key, which it can 
# .. use to obtain identity about the batch of runs in was in

import subprocess
import sys
import uuid

from tqdm import tqdm

assert len(sys.argv) in [3,4]

target_script = str(sys.argv[1])
number_of_times = int(sys.argv[2])

# Generate unique key for this set of runs for grouping
if len(sys.argv) == 4:
    tag = sys.argv[3]
else:
    tag = ''

key = tag + ' ' + str(uuid.uuid4())

for t in tqdm(range(number_of_times)):
    # subprocess.run(['source', '../../../4y'])
    subprocess.run(['python3', target_script, key])