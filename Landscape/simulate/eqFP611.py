import os
import pickle
import numpy as np
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt


class AdaptiveWalkSimulation:
    def __init__(self, population_size, loci, allele_count, fitness_table, mutation_rate):
        self.population_size = population_size
        self.loci = loci
        self.allele_count = allele_count
        self.fitness_table = fitness_table
        self.mutation_rate = mutation_rate

    def get_fitness(self, genotype):
        index = tuple(int(g) for g in genotype)
        return self.fitness_table[index]

    def generate_all_possible_mutations(self, genotype):
        mutations = []
        for i in range(self.loci):
            for new_allele in range(self.allele_count[i]):
                if new_allele != genotype[i]:
                    mutated_genotype = genotype.copy()
                    mutated_genotype[i] = new_allele
                    mutations.append(mutated_genotype)
        return mutations

    def calculate_transition_probability(self, current_genotype, new_genotype):
        current_fitness = self.get_fitness(current_genotype)
        new_fitness = self.get_fitness(new_genotype)
        s_ij = new_fitness - current_fitness

        if s_ij == 0:
            return 1 / self.population_size
        return max(0., (1 - np.exp(-2 * s_ij)) / (1 - np.exp(-2 * self.population_size * s_ij)))

    def run(self, initial_genotype, steps, seed=None):
        if seed is not None:
            np.random.seed(seed)

        current_genotype = initial_genotype
        history = [current_genotype]

        for _ in range(steps):
            mutation_occurs = np.random.rand() < (self.mutation_rate * self.population_size)
            if mutation_occurs:
                possible_mutations = self.generate_all_possible_mutations(current_genotype)
                probabilities = [self.calculate_transition_probability(current_genotype, mut) for mut in possible_mutations]
                
                if np.all(np.array(probabilities) == 0.):
                    import ipdb; ipdb.set_trace()
                
                probabilities = np.array(probabilities) / np.sum(probabilities)
                next_genotype = possible_mutations[np.random.choice(len(possible_mutations), p=probabilities)]
                current_genotype = next_genotype

            history.append(current_genotype)
        
        return history
    
    def get_optimal_genotype(self):
        optimal_genotype = max(self.fitness_table, key=self.fitness_table.get)
        return list(optimal_genotype), self.fitness_table[optimal_genotype]

    def find_local_optima(self):
        local_optima = []
        for genotype in self.fitness_table.keys():
            is_local_optima = True
            current_fitness = self.get_fitness(list(genotype))
            possible_mutations = self.generate_all_possible_mutations(list(genotype))
            
            for mutated_genotype in possible_mutations:
                if self.get_fitness(mutated_genotype) > current_fitness:
                    is_local_optima = False
                    break
            
            if is_local_optima:
                local_optima.append(list(genotype))
        
        return local_optima

def getSimTraj(conf, vis=True, save=True, traj_num=None, force=False):
    if os.path.exists(conf.data_dir + "evolution_trajectories.npy") and not force:
        return
    
    loci, states, steps = conf.eqFP611.loci, conf.eqFP611.states, conf.eqFP611.steps
    traj_num = traj_num or conf.eqFP611.traj_num # if traj_num is None, use conf.eqFP611.traj_num
    fitness_table = np.load(conf.data_dir + "preprocessed_fitness_table.npy", allow_pickle=True)
    
    np.random.seed(42)
    initial_genotype = [list(np.random.choice(range(states), size=loci)) for _ in range(traj_num)]
    
    system = AdaptiveWalkSimulation(
        loci=loci,
        allele_count=[states] * loci,
        population_size=conf.eqFP611.population_size,
        mutation_rate=conf.eqFP611.mutation_rate,
        fitness_table=fitness_table,
    )

    all_trajectories = []
    for idx in tqdm(range(traj_num), desc="Simulating Trajectories"):
        evolution_history = system.run(initial_genotype=initial_genotype[idx], steps=steps, seed=idx)[:steps]
        all_trajectories.append(evolution_history)
    
    if vis:
        plt.figure(figsize=(6, 3))
        for trajectory in all_trajectories[:1000]:
            plt.plot(range(steps), [system.get_fitness(genotype) for genotype in trajectory], alpha=0.5)
        plt.xlabel('Mutation')
        plt.ylabel('Fitness')
        plt.tight_layout()
        plt.savefig(conf.data_dir + "fitness_trajectories.png", dpi=300)
        
        # fig = plt.figure(figsize=(6, 6))
        # ax = fig.add_subplot(111, projection='3d')
        # for trajectory in all_trajectories[::25]:
        #     ax.plot(*zip(*trajectory), alpha=0.025)
        # ax.set_xlabel('Locus 1')
        # ax.set_ylabel('Locus 2')
        # ax.set_zlabel('Locus 3')
        # plt.tight_layout()
        # plt.savefig(conf.data_dir + "evolution_trajectories.png", dpi=600)
    
    if save:
        np.save(conf.data_dir + "evolution_trajectories.npy", all_trajectories)
        with open(conf.data_dir + 'system.pkl', 'wb') as f:
            pickle.dump(system, f)
    
    return all_trajectories

class Dataset_eqFP611(Dataset):
    def __init__(self, conf, traj=None):
        super().__init__()
        self.conf = conf

        try:
            self.load_data(traj)
        except:
            getSimTraj(conf)
            self.load_data(traj)

    def load_data(self, traj):
        if traj is None:
            self.traj = np.load(self.conf.data_dir + 'evolution_trajectories.npy', allow_pickle=True) # (N, T, L)
        else:
            self.traj = traj

        self.nstates = self.conf.eqFP611.states
        
        self.traj_onehot = np.eye(self.nstates)[self.traj.astype(int)]  # (N, T, L, nstates)

        self.X = self.traj_onehot.astype(np.float32)  # (N, T, L, nstates)
        self.X_label = self.traj.astype(int) # (N, T, L)

    def __getitem__(self, index):
        return self.X[index], self.X_label[index] 

    def __len__(self):
        return len(self.traj)
    
    def getLoader(self, batch_size, shuffle=True, drop_last=True):
        return DataLoader(self, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)


if __name__ == '__main__':
    
    from omegaconf import OmegaConf
    conf = OmegaConf.load('config/eqFP611.yaml')
    trajectorys = getSimTraj(conf, vis=True, save=False, traj_num=100, force=True)