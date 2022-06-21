import pandas as pd
import random

# Random instance generator
nr_instances = 13
# Define general file name for instances
destination = "instances/FJSP_"

# DSM Plant Input
product_types = ["enzyme0", "enzyme1", "enzyme2", "enzyme3", "enzyme4", "enzyme5"]
nr_products = len(product_types)
nr_units = 3    # preparation, filtering, reception

recipes = [[0, 1, 2],  # enzyme 0
           [0, 1],     # enzyme 1
           [1, 2],     # enzyme 2
           [0, 1, 2],  # enzyme 3
           [0, 1, 2],  # enzyme 4
           [1, 2]]     # enzyme 5

machines = [0, 1, 2, 3, 4, 5, 6, 7, 8]

unitMachines = {
            0: [0, 1, 2],  # machines for preparation
            1: [3, 4, 5, 6],  # machines for filtering
            2: [7, 8]}  # machines for reception

processing_times = [[8, 3, 0, 4, 5, 0],  # preparation, [enzyme0, enzyme1, enzyme2, enzyme3, enzyme4, enzyme5]
                    [4, 2, 3, 6, 4, 8],  # filtering  , [enzyme0, enzyme1, enzyme2, enzyme3, enzyme4, enzyme5]
                    [4, 0, 3, 6, 7, 3]]  # reception  , [enzyme0, enzyme1, enzyme2, enzyme3, enzyme4, enzyme5]

# Read change over matrix from csv
change_overs_csv = pd.read_csv("instances/input/change_overs.csv")
change_overs = {}
for index, row in change_overs_csv.iterrows():
    change_overs[(row['Machine'], row['Product1'], row['Product2'])] = row["ChangeOver"]

operations = {}
machineAlternatives = {}
processingTimes = {}

for inst in range(nr_instances):
    orders = {}
    job_ID = 0
    for product in range(nr_products):
        nr_jobs = 1 + inst # define distribution for nr of batches per enzyme
        for i in range(nr_jobs):
            orders[job_ID] = {"product": product_types[product], "due": 10 + random.randint(0, (1+inst) * 10)}
            recipe = recipes[product]
            nr_operations = len(recipe)
            operations[job_ID] = [i for i in range(len(recipe))]
            for j in operations[job_ID]:
                unit = recipe[j]
                alternatives = unitMachines[unit]
                machineAlternatives[job_ID,j] = unitMachines[unit]
                for k in alternatives:
                    processingTimes[job_ID, j, k] = processing_times[unit][product]
            job_ID += 1

    jobs = [i for i in range(job_ID)]

    with open(destination + str(inst) +'.py', 'w') as f:
        f.write("nr_machines = ")
        f.write(str(len(machines)))
        f.write('\n')
        f.write("nr_jobs = ")
        f.write(str(len(jobs)))
        f.write('\n')
        f.write("orders = ")
        f.write(str(orders))
        f.write('\n')
        f.write("machines = ")
        f.write(str(machines))
        f.write('\n')
        f.write("jobs = ")
        f.write(str(jobs))
        f.write('\n')
        f.write("operations = ")
        f.write(str(operations))
        f.write('\n')
        f.write("machineAlternatives = ")
        f.write(str(machineAlternatives))
        f.write('\n')
        f.write("processingTimes = ")
        f.write(str(processingTimes))
        f.write('\n')
        f.write("changeOvers = ")
        f.write(str(change_overs))
        f.write('\n')
        f.close()