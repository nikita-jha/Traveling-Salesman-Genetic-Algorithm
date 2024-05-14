import random
import numpy as np
from deap import base, creator, tools, algorithms
import math
from deap.tools.support import HallOfFame

city_list = {}
with open('TSPDATA.txt', 'r') as f:
    lines = f.readlines()[2:]
    for l in lines:
        data = l.split()
        city_list[int(data[0])] = (int(data[1]), int(data[2]))

creator.create("FitnessMin", base.Fitness, weights=(-1,))
creator.create("Individual", list, fitness=creator.FitnessMin)
toolbox = base.Toolbox()
toolbox.register("indices", random.sample, range(0, 127), 127)  # Start at 0, end at 12 for 127 cities
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.indices)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

#Calculate Euclidean Distance
def distance(city1, city2):
    x1, y1 = city1
    x2, y2 = city2
    x_diff = x2 - x1
    y_diff = y2 - y1
    dist = math.sqrt((x_diff)**2 + (y_diff)**2)
    return dist

#Fitness function
def evalTPSSolution(individual):
    # Adjust indices to be 1-based for evaluation since cities dictionary starts at 1
    adjusted_individual = [city+1 for city in individual]
    total_sum = 0
    for i in range(len(adjusted_individual)):
        dist = distance(city_list[adjusted_individual[i]], city_list[adjusted_individual[(i+1)%len(adjusted_individual)]])
        total_sum += dist
    return (total_sum,)


toolbox.register("evaluate", evalTPSSolution)
toolbox.register("mate", tools.cxOrdered)
toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.01)  
toolbox.register("select", tools.selTournament, tournsize=15)  

def main():
    pop = toolbox.population(n=17000)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("Min", np.min)
    base_cxpb, base_mutpb = 0.8, 0.2
    for gen in range(250):
        mutpb = base_mutpb + (1 - base_mutpb) * (gen / 250)
        cxpb = base_cxpb - (base_cxpb * 0.5) * (gen / 250)  
        offspring = algorithms.varAnd(pop, toolbox, cxpb=cxpb, mutpb=mutpb)
        fitnesses = toolbox.map(toolbox.evaluate, offspring)
        
        for fitness, individual in zip(fitnesses, offspring):
            individual.fitness.values = fitness
        
        hof.update(offspring)
        pop = toolbox.select(offspring, 17000 - 50)  
        pop += tools.selBest(offspring, 50) 
        
        print(f"Best Fitness in Generation {gen}: {hof.items[0].fitness.values[0]}")
    
    best_individual = hof.items[0]
    print('Final Best Individual: ', best_individual)
    print('Final Best Fitness: ', best_individual.fitness.values[0])

if __name__ == "__main__":
    main()
