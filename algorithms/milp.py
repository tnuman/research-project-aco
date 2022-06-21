from gurobipy import *
import pandas as pd


class FlexibleJobShop:
    def __init__(
            self, jobs, orders, machines, processingTimes, operations, machineAlternatives, instance, changeOvers):
        self.model = Model("FJSP")
        self.jobs = jobs
        self.orders = orders
        self.changeOvers = changeOvers
        self.machines = machines
        self.processingTimes = processingTimes
        self.operations = operations
        self.machineAlternatives = machineAlternatives
        self.L = 1000
        self.instance = instance

    def build_model(self, csv_output, time=0):

        if time != 0:
            self.model.setParam('TimeLimit', time)

        # Sets
        J = set([(i) for i in self.jobs])
        JOM = set([(i, j, k) for i in self.jobs for j in self.operations[i] for k in self.machineAlternatives[i, j]])
        JOJOM = set([(i, j, g, h, k) for i in self.jobs for j in self.operations[i] for g in self.jobs for h in self.operations[g] for k in
                     self.machineAlternatives[i, j] for l in self.machineAlternatives[g, h] if i != g and k == l])

        # Variables
        # x[i,j, m]=1 if operation j of job i is scheduled on machine k
        x = self.model.addVars(JOM, vtype=GRB.BINARY, name='x')

        # Create variables for start times for each operation of each job at each machine
        s = self.model.addVars(JOM, vtype=GRB.INTEGER, name='s', lb=0)

        # Create variables for completion times for each operation of each job at each machine
        c = self.model.addVars(JOM, vtype=GRB.INTEGER, name='c', lb=0)

        # Create variables for completion times for each job
        f = self.model.addVars(J, vtype=GRB.INTEGER, name='f', lb=0)

        # y[i,j,g,h,k] = 1 if operation (i,j) precedes operation (g,h) on machine k
        y = self.model.addVars(JOJOM, vtype=GRB.INTEGER, name='y')

        # The objective will be the makespan
        Cmax = self.model.addVar(obj=1, vtype=GRB.INTEGER, name='z')

        # Constraints
        #  1. Each operation only gets one assignment
        self.model.addConstrs((quicksum([x[i, j, k] for k in self.machineAlternatives[i, j]]) == 1 for i in self.jobs
                               for j in self.operations[i]),'oneAssignment')

        # 2. Start time and completion time are set to 0 if not chosen to do at machine k
        self.model.addConstrs((s[i, j, k] + c[i, j, k] <= x[i, j, k] * self.L for (i, j, k) in JOM), "notChosen")

        # 3. Disjunctive constraint  guarantee that the difference between the starting and the completion times
        # is equal in the least to the processing time on machine
        self.model.addConstrs(
            (c[i, j, k] >= s[i, j, k] + self.processingTimes[i, j, k] - self.L * (1 - x[i, j, k]) for (i, j, k) in JOM),
            "startCompletion")

        # 4 Two jobs cannot be done at one machine at the same time
        self.model.addConstrs((s[i, j, k] >= c[g, h, k] + self.changeOvers[k, self.orders[g]['product'], self.orders[i]['product']] -
                               (2 - x[i,j,k] - x[g,h,k] + y[i, j, g, h, k]) * self.L for (i, j, g, h, k) in JOJOM), "oneOperationPerMachine")

        # 5 Two jobs cannot be done at one machine at the same time (disjunctive)
        self.model.addConstrs((s[g, h, k] >= c[i, j, k] + self.changeOvers[k, self.orders[i]['product'], self.orders[g]['product']] - (3 - x[i,j,k] - x[g,h,k]
                            -y[i, j, g, h, k]) * self.L for (i, j, g, h, k) in JOJOM), "disjunctiveOneOperation")

        # 6. Ensures that precedence relations between two operations are not violated
        self.model.addConstrs((quicksum([s[i, j, k] for k in self.machineAlternatives[i, j]]) >=
                               quicksum([c[i, j - 1, k] for k in self.machineAlternatives[i, j - 1]]) for i in
                               self.jobs for j in
                               self.operations[i] if j > 0), 'precedingOperations')

        # 7. Determine completion time of jobs
        self.model.addConstrs(
            (f[i] == quicksum([c[i, self.operations[i][-1], k] for k in self.machineAlternatives[i, self.operations[i][-1]]]) for i
             in J), 'finishJob')

        # 8.  Minimize make span, make span is bigger than all finish times of jobs
        self.model.addConstrs((f[i] <= Cmax for i in J), 'objectiveConstraint')

        self.model.update()
        self.model.write('solutions/milp/models/' + self.instance + '.lp')
        self.model.optimize()

        s_m = dict(self.model.getAttr('X', s))
        x_m = dict(self.model.getAttr('X', x))

        """ Write results to csv"""
        results = []

        for i in self.jobs:
            for j in self.operations[i]:
                for k in self.machineAlternatives[i, j]:
                    if x_m[i, j, k] > 0.5:
                        results.append({"Machine": k, "Job": i, "Product": self.orders[i]["product"], "Operation": j,  "Start": s_m[i, j, k], "Duration":
                            self.processingTimes[i, j, k], "Completion": s_m[i, j, k] + self.processingTimes[i, j, k]})

        if self.model.status == GRB.OPTIMAL:
            print('The make span is ', self.model.objVal)
        else:
            print("Could not solve to optimality")

        schedule = pd.DataFrame(results)
        schedule.to_csv(csv_output, index=False)

        return schedule