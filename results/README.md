# Results

## Weights and Biases set up (for logging)
Weights and Biases (wandb) is a machine learning logging tool similar to tensorboard, except it stores your data in the cloud and makes it shareable to others.

wandb should be installed from requirements.txt. If not ```pip install wandb``` to install it.  
```wandb login``` to sign up and log in to a weights and bias account.

Thats it! Running the below scripts should now log to your wandb account.  
To share - set the project to public and share the link. (more private sharing requires a subscription).

For more information about how wandb works and how we log to it spend 5 minutes reading their quickstart here: https://docs.wandb.com/quickstart

## About these scripts
  
All the scripts in this folder, bar runTestMultipleTimes.py, follow the same structure:  
  
1. import a problem from problems
2. define an integrator class as seen in the example notebooks
3. define an experiment that tests the integrator on the problem
4. define a main function that runs the experiment on multiple dimensions when the script is invoked.

To run the a script, such as ```simpleIntegrator.py``` run  
```python3 simpleIntegrator.py <tag>``` where <tag> is passed to the wandb run info dictionary to allow multiple runs to be grouped together.
  
  
## Running a test multiple times
runTestMultipleTimes.py does this. When invoked as 
`python3 runTestMultipleTimes.py simpleIntegrator.py 30 <groupTag>` , it runs the script simpleIntegrator.py 30 times, all with the tag <groupTag>. On wandb, we can then group all these runs together using this tag, and produce the error plots.
