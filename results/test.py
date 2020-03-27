import argparse
import wandb

wandb.init(project='BoilerPlate')

parser = argparse.ArgumentParser()
parser.add_argument('--N')
args = parser.parse_args()

print(vars(args))
print(vars(args)['N'])
print(type(wandb.config))
print(wandb.config)
