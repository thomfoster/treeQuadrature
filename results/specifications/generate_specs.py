import yaml

defaults = {
    'verbose': True,
    'num_cores': 6,
    'num_runs': 6,
    'key': 'by_target_name',
    'remove_nonce': False,
    'wandb_project': 'by_problem_name',
    'problem': 'SimpleGaussian'  # Lets have 1 script per problem
}

experiments = []


###########################
# Smc Experiments
###########################

experiments.append({
    "target": "test_smcIntegrator.py",
    "N": 20_00
})


############################
# Vegas Experiments
############################

target = "test_vegasIntegrator.py"

N_NITN_PAIRS = [
    # (1_00, 20),
    (2_00, 10),
    # (5_00, 4),
    # (10_00, 2)
]

for N, NITN in N_NITN_PAIRS:
    experiments.append({
        "target": target,
        "N": N,
        "NITN": NITN
    })


###############################
# Simple Integrator Experiments
################################

target = "test_simpleIntegrator.py"

splits = [
    "kdSplit",
    "minSseSplit"
]

Ps = [
    2, # 5, 10, 20
]

for split in ['kdSplit']:
    for P in Ps:
        experiments.append({
            'target': target,
            'N': 20_00,
            'P': P,
            'split': split,
            'integral': 'midpointIntegral'
        })

N_extra_N_pairs = [
    # (20_00, 1),
    (10_00, 2),
    # (5_00, 4),
    (2_00, 10),
    # (20, 100),
    # (2, 1000)
]

integrals = [
    'smcIntegral',
    # 'randomIntegral'
]

for split in splits:
    for N, extra_N in N_extra_N_pairs:
        for integral in integrals:
            experiments.append({
                'target': target,
                'N': N,
                'P': 2,
                'split': split,
                'integral': integral,
                'num_extra_samples': extra_N
            })


##########################################
# Limited Sample Integrator Experiments
###########################################

target = 'test_limitedSampleIntegrator.py'

base_Ns = [
    # 1.0,
    0.75,
    # 0.5,
    # 0.25,
    0.1
]

splits = [
    'kdSplit',
    'minSseSplit'
]

extra_Ns = [
    4, 20
]

weighting_functions = [
    'yvar', 'range', 'volume'
]

for base_N in base_Ns:
    for split in splits:
        for extra_N in extra_Ns:
            for weighting_function in weighting_functions:
                N = int(20_00 / extra_N)
                actual_base_N = int(base_N * N)
                experiments.append({
                    'target': target,
                    'N': N,
                    'base_N': actual_base_N,
                    'active_N': 5,
                    'split': split,
                    'integral': 'smcIntegral',
                    'num_extra_samples': extra_N, 
                    'weighting_function': weighting_function,
                    'queue': 'PriorityQueue'
                })


#########################
# Push into yaml
#########################

yaml_dict = {
    'defaults': defaults,
    'experiments': experiments
}

print(f"Generated {len(experiments)} experiments")

with open('specifications_generated.yaml', 'w') as f:
    yaml.dump(yaml_dict, f)