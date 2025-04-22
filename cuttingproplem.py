import random
import time
import numpy as np



#--------------------------------
#Simulated Annealing
#-------------------------------


class SA_Agent:
    def __init__(self, fabric_length, piece_list):
        self.fabric_length = fabric_length
        self.piece_list = piece_list

    def waste(self, pieces):

        total_length = sum(pieces)
        return max(self.fabric_length - total_length, 0)

    def neighbour_solution(self, pieces):

        new_pieces = pieces[:]
        i, j = random.sample(range(len(new_pieces)), 2)
        new_pieces[i], new_pieces[j] = new_pieces[j], new_pieces[i]
        return new_pieces

    def neighbour_solution2(self, pieces):

        new_pieces = pieces[:]
        i, j, k = random.sample(range(len(new_pieces)), 3)
        new_pieces[i], new_pieces[j], new_pieces[k] = new_pieces[k], new_pieces[j], new_pieces[i]
        return new_pieces

    def count_pieces_in_one_fabric(self,pieces, fabric_length):
        total = 0
        counter = 0
        for p in pieces:
            if total + p <= fabric_length:
                total += p
                counter += 1
            else:
                break
        return counter

    def control(self, cost_difference, T):

        q = random.random()
        return q <= np.exp(-cost_difference / T)

    def optimize(self, cool_rate=0.95, initial_temp=100):

        start_time = time.time()
        T = initial_temp
        pieces = self.piece_list[:]
        random.shuffle(pieces)

        best_solution = pieces[:]
        best_cost = self.waste(pieces)

        all_costs = [best_cost]

        while T > 0.01:
            current_solution = self.neighbour_solution(pieces)
            current_cost = self.waste(current_solution)

            # optimization with neighbours
            for _ in range(5):
                new_solution = self.neighbour_solution2(pieces)
                new_cost = self.waste(new_solution)
                if new_cost < current_cost:
                    current_solution = new_solution
                    current_cost = new_cost

            cost_diff = current_cost - best_cost

            if cost_diff < 0 or self.control(cost_diff, T):

                pieces = current_solution
                if current_cost < best_cost:
                    best_solution = current_solution
                    best_cost = current_cost

            all_costs.append(best_cost)
            T *= cool_rate  # Sıcaklığı azalt
            end_time= time.time()
            total_time = end_time - start_time

        return best_solution, best_cost,total_time


#--------------------------------
#HILL CLIMBING
#-------------------------------

class HC_Agent:
    def __init__(self, fabric_length, piece_list):
        self.fabric_length = fabric_length
        self.piece_list = piece_list

    def waste(self, pieces):
        total = 0
        for p in pieces:
            if total + p <= self.fabric_length:
                total += p
            else:
                break
        return self.fabric_length - total

    def neighbour_solution(self, pieces):
        new_pieces = pieces[:]
        i, j = random.sample(range(len(new_pieces)), 2)
        new_pieces[i], new_pieces[j] = new_pieces[j], new_pieces[i]
        return new_pieces

    def count_pieces_in_one_fabric(self,pieces, fabric_length):
        total = 0
        counter = 0
        for p in pieces:
            if total + p <= fabric_length:
                total += p
                counter += 1
            else:
                break
        return counter

    def optimize(self):
        start_time = time.time()
        random.shuffle(self.piece_list)
        current_solution=self.piece_list[:]
        best_cost = self.waste(current_solution)
        best_solution=current_solution
        improved=True

        while improved:
            improved = False
            neighbor = self.neighbour_solution(current_solution)
            cost = self.waste(neighbor)
            if cost < best_cost:
                best_cost = cost
                current_solution = neighbor
                best_solution = neighbor
                improved = True

        end_time = time.time()
        total_time = end_time - start_time
        return best_solution, best_cost, total_time

class GA_Agent:
    def __init__(self, fabric_length, piece_list, population_size=50, generations=100, mutation_rate=0.1):
        self.fabric_length = fabric_length
        self.piece_list = piece_list
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate

    def waste(self, pieces):
        total = 0
        for p in pieces:
            if total + p <= self.fabric_length:
                total += p
            else:
                break
        return self.fabric_length - total

    def fitness(self, chromosome):
        return self.fabric_length - self.waste(chromosome)

    def initialize_population(self):
        population = []
        for _ in range(self.population_size):
            individual = self.piece_list[:]
            random.shuffle(individual)
            population.append(individual)
        return population

    def selection(self, population):
        tournament = random.sample(population, 5)
        tournament.sort(key=lambda x: self.fitness(x), reverse=True)
        return tournament[0]

    def crossover(self, parent1, parent2):
        size = len(parent1)
        start, end = sorted(random.sample(range(size), 2))
        child = [None] * size
        child[start:end] = parent1[start:end]
        pointer = 0
        for gene in parent2:
            if gene not in child:
                while child[pointer] is not None:
                    pointer += 1
                child[pointer] = gene

        for i in range(size):
            if child[i] is None:
                child[i] = random.choice(parent1)
        return child

    def mutate(self, chromosome):
        if random.random() < self.mutation_rate:
            i, j = random.sample(range(len(chromosome)), 2)
            chromosome[i], chromosome[j] = chromosome[j], chromosome[i]

    def count_pieces_in_one_fabric(self, pieces, fabric_length):
        total = 0
        counter = 0
        for p in pieces:
            if total + p <= fabric_length:
                total += p
                counter += 1
            else:
                break
        return counter
    def optimize(self):
        start_time = time.time()
        population = self.initialize_population()
        best_solution = max(population, key=self.fitness)

        for _ in range(self.generations):
            new_population = []
            for _ in range(self.population_size):
                parent1 = self.selection(population)
                parent2 = self.selection(population)
                child = self.crossover(parent1, parent2)
                self.mutate(child)
                new_population.append(child)

            population = new_population
            current_best = max(population, key=self.fitness)
            if self.fitness(current_best) > self.fitness(best_solution):
                best_solution = current_best

        end_time = time.time()
        best_cost = self.waste(best_solution)
        return best_solution, best_cost, end_time - start_time

#--------------------------------
# Collaborative Optimization
#-------------------------------

def collaborative_optimization(agents):
    best_solutions = []
    for agent in agents:
        pattern, waste, time_taken = agent.optimize()
        best_solutions.append((pattern, waste, time_taken))

    best_overall = min(best_solutions, key=lambda x: x[1])  # least waste
    return best_overall

#--------------------------------
# Hyper Meta-Heuristic Approach
#-------------------------------

def hyper_meta_heuristic(sa, hc, ga):
    results = [
        ("SA", *sa.optimize()),
        ("HC", *hc.optimize()),
        ("GA", *ga.optimize())
    ]
    best = min(results, key=lambda x: x[1])  # waste
    return best

#--------------------------------
# Menu
#-------------------------------
# Kullanıcıdan girdileri al
fabric_length = int(input("Enter the length of the fabric: "))
piece_lengths = list(map(int, input("Enter the required piece lengths, separated by spaces: ").split()))
quantity = list(map(int, input("Enter the required quantity, separated by spaces: ").split()))

# Parça uzunluklarını uygun miktarlarda çoğalt
full_piece_list = []
for length, qty in zip(piece_lengths, quantity):
    full_piece_list.extend([length] * qty)

num_agents = input("Enter number of agents (1 for SA, 2 for HC, 3 for GA, or all for hyper-meta-heuristic): ")
if num_agents == '1':
    sa_agent = SA_Agent(fabric_length, full_piece_list)
    best_pattern, best_waste, total_time = sa_agent.optimize()
    used_pieces = sa_agent.count_pieces_in_one_fabric(best_pattern, fabric_length)
    used_length = sum(best_pattern[:used_pieces])
    remaining_fabric = fabric_length - used_length
    print(f"Simulated Annealing:")
    print(f"Execution time: {total_time}")
    print(f"best pattern: {best_pattern}")
    print(f"number of pieces: {used_pieces}")
    print(f"Remaining Fabric in first roll: {remaining_fabric} metre")
elif num_agents == '2':
    hc_agent = HC_Agent(fabric_length, full_piece_list)
    best_pattern2, best_waste2, total_time2 = hc_agent.optimize()
    used_pieces2 = hc_agent.count_pieces_in_one_fabric(best_pattern2, fabric_length)
    used_length2 = sum(best_pattern2[:used_pieces2])
    remaining_fabric2 = fabric_length - used_length2
    print(f"\nHill Climbing:")
    print(f"Execution time: {total_time2}")
    print(f"best pattern: {best_pattern2}")
    print(f"number of pieces: {used_pieces2}")
    print(f"Remaining Fabric in first roll: {remaining_fabric2} metre")
elif num_agents == '3':
    ga_agent = GA_Agent(fabric_length, full_piece_list)
    best_pattern3, best_waste3, total_time3 = ga_agent.optimize()
    used_pieces3 = ga_agent.count_pieces_in_one_fabric(best_pattern3, fabric_length)
    print(f"\nGenetic Algorithm:")
    print(f"Execution time: {total_time3:.4f} s")
    print(f"Best pattern: {best_pattern3}")
    print(f"Number of pieces used: {used_pieces3}")
    print(f"Remaining fabric in first roll: {best_waste3} metre")
elif num_agents == 'all':
    agents = [
        SA_Agent(fabric_length, full_piece_list),
        HC_Agent(fabric_length, full_piece_list),
        GA_Agent(fabric_length, full_piece_list)
    ]
    best_pattern, best_waste, best_time = collaborative_optimization(agents)


    print(f"\nCollaborative Optimization Result: {best_pattern}")
    print(f"\n Waste: {best_waste}")
    print(f"\n Time: {best_time}")

    sa_agent = SA_Agent(fabric_length, full_piece_list)
    hc_agent = HC_Agent(fabric_length, full_piece_list)
    ga_agent = GA_Agent(fabric_length, full_piece_list)
    best_algorithm, best_pattern, best_waste, best_time = hyper_meta_heuristic(sa_agent, hc_agent, ga_agent)
    print(f"\nHyper Meta-Heuristic Approach:")
    print(f"Best Algorithm: {best_algorithm}")
    print(f"Best Pattern: {best_pattern}")
    print(f"Waste: {best_waste}")
    print(f"Execution Time: {best_time}")
else:
    print("Invalid input")
