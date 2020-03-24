# Obtaining Results
  
All the scripts in this folder, bar runTestMultipleTimes.py, follow the same structure:  
  
1. import a problem from problems
2. define an integrator class
3. define an experiment that tests the integrator on the problem
4. define a main function that runs the experiment on multiple dimensions when the script is invoked.
  
runTestMultipleTimes.py does exactly what it says. When invoked as 
`python3 runTestMultipleTimes.py simpleIntegrator.py 30` , it runs the script simpleIntegrator.py 30 times.
The trick is that it passes a common "key" to each run, that that run logs to wandb.com.
  
This allows us to group the runs together, and produce error estimates.