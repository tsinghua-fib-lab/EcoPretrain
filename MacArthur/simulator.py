import os
import numpy as np
from tqdm import tqdm


def itoEuler(f, y0, tspan):
    """dy = f(y,t)dt

    where y is the d-dimensional state vector, f is a vector-valued function

    Args:
      f: callable(y, t) returning (d,) array
         Vector-valued function to define the deterministic part of the system
      y0: array of shape (d,) giving the initial state vector y(t==0)
      tspan (array): The sequence of time points for which to solve for y.
        These must be equally spaced, e.g. np.arange(0,10,0.005)
        tspan[0] is the intial time corresponding to the initial state y0.

    Returns:
      y: array, with shape (len(tspan), len(y0))
         With the initial value y0 in the first row
    """
    N = len(tspan)
    h = (tspan[N-1] - tspan[0])/(N - 1)
    d = len(y0)
    # allocate space for result
    y = np.zeros((N, d), dtype=y0.dtype)
    y[0] = y0
    for n in range(0, N-1):
        tn = tspan[n]
        yn = y[n]
        y[n+1] = yn + f(yn, tn)*h
    return y


class EcoSimulator:
    
    def __init__(self, n_traits, num_resources, trait_inter_matrix):
        
        # Species
        self.n_traits = n_traits # `sigma`: n_species x num_resources
        self.trait_inter_matrix = trait_inter_matrix # `J`: num_resources x num_resources
        self.n_abundances = np.array([]) # `N`: n_species x timesteps
        
        # Resoureces
        self.num_resources = num_resources
        self.max_benefit = np.ones(num_resources) #`b`: num_resources
        self.carrying_capacity = np.ones(num_resources) * 1e3 # `K`: num_resources
        
        # Simulation parameters
        self.c = 0.1 # baseline cost
        self.ksi = 0.5 # trait carrying cost
        
    def resource_benefit(self, n_abundances, n_traits, n_active_mask):
        exploitation = np.dot(n_traits.T, n_abundances * n_active_mask)
        n_benefits = self.max_benefit / (1 + exploitation / self.carrying_capacity)
        return n_benefits
        
    def growth_rate(self, n_abundances, n_traits, n_active_mask):
        n_benefits = self.resource_benefit(n_abundances, n_traits, n_active_mask)
        active_n_traits = np.einsum('ij,i->ij', n_traits, n_active_mask)
        n_growth_rate = np.dot(active_n_traits, n_benefits)
        return n_growth_rate
    
    def survival_cost(self, n_traits, n_active_mask):
        baseline_cost = self.c
        trait_carrying_cost = self.ksi * np.sum(n_traits, axis=-1)
        active_n_traits = np.einsum('ij,i->ij', n_traits, n_active_mask)
        interaction_cost = np.sum(active_n_traits * np.dot(active_n_traits, self.trait_inter_matrix), axis=-1)
        n_survival_cost = (baseline_cost + trait_carrying_cost) * n_active_mask - interaction_cost
        return n_survival_cost
    
    def get_fitness(self, n_abundances, n_traits, n_active_mask):
        return self.growth_rate(n_abundances, n_traits, n_active_mask) - self.survival_cost(n_traits, n_active_mask)
    
    def get_mutation_fitness(self, parents_abundances, parents_traits, candidates_traits, parents_active_mask):
        n_benefits = self.resource_benefit(parents_abundances, parents_traits, parents_active_mask)
        n_growth_rate = np.dot(candidates_traits, n_benefits)
        return n_growth_rate
    
    def dynamics(self, n_abundances, t):
        n_active_mask = (n_abundances / n_abundances.sum()) > 1e-6
        dNdt = n_active_mask * self.get_fitness(n_abundances, self.n_traits, n_active_mask)
        return dNdt
    
    def mutation_candidates(self, n_traits, n_active_mask):
        """
        Generate mutation candidates, including previously extinct species.
        """
        active_n_traits = n_traits[n_active_mask]
        n_parents = active_n_traits.shape[0]
        mutation_candidates = np.zeros((n_parents, self.num_resources, self.num_resources))
        extinct_mask = np.zeros((n_parents, self.num_resources))
        for i in range(n_parents):
            for j in range(self.num_resources):
                mutant = active_n_traits[i].copy()
                mutant[j] = 1 - mutant[j]
                mutation_candidates[i, j] = mutant  # No need to check for existing species
        
        return mutation_candidates, extinct_mask
    
    def mutation_event(self):
        """
        Mutation event with reappearance of extinct species by updating existing abundance.
        """
        n_active_mask = (self.n_abundances[-1] / self.n_abundances[-1].sum()) > 1e-6
        candidates, extinct_mask = self.mutation_candidates(self.n_traits, n_active_mask)
        
        # Calculate mutation probability (similar to before)
        fitness = np.ones((candidates.shape[0], candidates.shape[1]))
        mutation_prob = np.zeros((candidates.shape[0] * self.num_resources))
        for i in range(candidates.shape[0]):
            for j in range(self.num_resources):
                mutation_prob[i * self.num_resources + j] = fitness[i, j] * self.n_abundances[-1, n_active_mask][i]
        mutation_prob = mutation_prob / mutation_prob.sum()

        # Mutation species
        mutation_idx = np.random.choice(candidates.shape[0] * self.num_resources, p=mutation_prob)
        mutation_trait = candidates.reshape(-1, self.num_resources)[mutation_idx]
        
        # Check if the mutant trait already exists in `self.n_traits`
        existing_index = np.where((self.n_traits == mutation_trait).all(axis=1))[0]
        if existing_index.size > 0:
            # Mutant trait exists, update abundance at the existing index
            extinct_abundance_index = existing_index[0]
            self.n_abundances[-1, extinct_abundance_index] += max(1 / fitness.flatten()[mutation_idx], 1)
        else:
            # Add new trait and initialize abundance if trait is truly new
            self.n_traits = np.vstack([self.n_traits, mutation_trait])
            new_species_abundance = np.zeros(self.n_abundances.shape[0])
            new_species_abundance[-1] = max(1 / fitness.flatten()[mutation_idx], 1)
            self.n_abundances = np.concatenate([self.n_abundances, new_species_abundance.reshape(-1, 1)], axis=1)

        return True
    
    def run(self, x0, total_t, dt, mutation_num, is_print=True):
        
        tspan = np.arange(0, total_t, dt)
        self.n_abundances = itoEuler(self.dynamics, np.array(x0), tspan)
        
        # mutation + simulation
        for i in range(mutation_num):
            # mutation event
            if not self.mutation_event():
                if is_print:
                    print(f'Mutation {i+1} failed!')
                break
            # simulation
            sol = itoEuler(self.dynamics, self.n_abundances[-1], tspan)
            self.n_abundances = np.concatenate([self.n_abundances, sol], axis=0)
        
        self.n_abundances = np.clip(self.n_abundances, 0, None)
        
        return self.n_abundances
    
    def run_one_round(self, x0, total_t, dt):
        tspan = np.arange(0, total_t, dt)
        n_abundances = itoEuler(self.dynamics, np.array(x0), tspan)
        n_abundances = np.clip(n_abundances, 0, None)
        return n_abundances


def reconsitution_test(strains_pool, num_resources, trait_inter_matrix, total_t, dt, repeat=100):

    final_abundances = np.zeros((repeat, len(strains_pool)))
    for i in range(repeat):
        init_traits = np.array([strains[np.random.randint(0, len(strains))] for strains in strains_pool])
        init_abundance = np.array([1.] * len(strains_pool))
        simulator = EcoSimulator(
            n_traits=init_traits,
            num_resources=num_resources,
            trait_inter_matrix=trait_inter_matrix,
        )
        final_abundances[i] = simulator.run_one_round(init_abundance, total_t, dt)[-1]
    
    relative_final_abundances = final_abundances / final_abundances.sum(axis=1, keepdims=True)
    relative_std = np.std(relative_final_abundances, axis=0)
    relative_mean = np.mean(relative_final_abundances, axis=0)
    
    # q_rec = (relative_mean * relative_std).sum()
    q_rec = relative_std.sum()
    return q_rec

def evolution_test(strains_pool, num_resources, trait_inter_matrix, total_t, dt, mutation_num, repeat=100):
    relative_final_abundance = np.zeros((repeat, len(strains_pool)))
    for i in tqdm(range(repeat)):
        init_traits = np.array([strains[np.random.randint(0, len(strains))] for strains in strains_pool])
        init_abundance = np.array([1.] * len(strains_pool)) 
        simulator = EcoSimulator(
            n_traits=init_traits,
            num_resources=num_resources,
            trait_inter_matrix=trait_inter_matrix,
        )
        final_abundances = simulator.run(init_abundance, total_t, dt, mutation_num=mutation_num)[-1]
        n_traits = simulator.n_traits
    
        for trait in n_traits:
            idx = [i for i, strains in enumerate(strains_pool) if np.any(np.all(strains == trait, axis=1))][0]
            relative_final_abundance[i, idx] += final_abundances[idx] / final_abundances.sum()
    
    relative_std = np.std(relative_final_abundance, axis=0)
    relative_mean = np.mean(relative_final_abundance, axis=0)
    
    q_evo = (relative_mean * relative_std).sum()
    # q_evo = relative_std.sum()
    return q_evo

def leave_one_out_test(strains_pool, num_resources, trait_inter_matrix, total_t, dt, repeat=100):
    relative_std = np.zeros((repeat, len(strains_pool)))
    relative_mean = np.zeros((repeat, len(strains_pool)))
    for i in range(repeat):
        for j in range(len(strains_pool)):
            init_traits = np.array([strains[np.random.randint(0, len(strains))] for k, strains in enumerate(strains_pool) if k != j])
            init_abundance = np.array([1.] * (len(strains_pool) - 1))
            
            metrices = []
            for specis in strains_pool[j]:
                init_traits = np.vstack([init_traits, specis])
                init_abundance = np.append(init_abundance, 1.)
                simulator = EcoSimulator(
                    n_traits=init_traits,
                    num_resources=num_resources,
                    trait_inter_matrix=trait_inter_matrix,
                )
                final_abundance = simulator.run_one_round(init_abundance, total_t, dt)[-1]
                metrices.append(final_abundance[-1] / final_abundance.sum())
            
            relative_std[i, j] = np.std(metrices)
            relative_mean[i, j] = np.mean(metrices)
    relative_std = relative_std.mean(axis=0)
    relative_mean = relative_mean.mean(axis=0)
    q_inv = (relative_mean * relative_std).sum()
    return q_inv


def invasion(init_traits, invasion_trait, num_resources, trait_inter_matrix, total_t, dt):
    init_abundance = np.array([1.] * len(init_traits)) 
    # simulator = EcoSimulator(
    #     n_traits=init_traits,
    #     num_resources=num_resources,
    #     trait_inter_matrix=trait_inter_matrix,
    # )
    # final_abundance = simulator.run_one_round(init_abundance, total_t, dt)[-1]
    
    init_traits_ = np.vstack([init_traits, invasion_trait])
    init_abundance_ = np.append(init_abundance, 1.)
    simulator = EcoSimulator(
        n_traits=init_traits_,
        num_resources=num_resources,
        trait_inter_matrix=trait_inter_matrix,
    )
    # final_abundance_ = simulator.run_one_round(init_abundance_, total_t, dt)[-1, :len(init_traits)]
    flag = simulator.run_one_round(init_abundance_, total_t, dt)[-1, -1] > 1e-6
    
    # diff = final_abundance_ - final_abundance
    
    return flag

def population_function(init_traits, num_resources, trait_inter_matrix, total_t, dt):
    init_abundance = np.array([1.] * len(init_traits))
    
    simulator = EcoSimulator(
        n_traits=init_traits,
        num_resources=num_resources,
        trait_inter_matrix=trait_inter_matrix,
    )
    final_abundance = simulator.run_one_round(init_abundance, total_t, dt)[-1]
    
    functional_value = final_abundance.sum()
    return functional_value


def process_abundance_history(abundance_history, trait_list, num_resources, max_mutations, time_interval, threshold=1e-6):
    num_species = 2**num_resources 
    
    result = np.full((max_mutations, num_species, num_resources), -1, dtype=int)
    
    for mutation_idx in range(max_mutations):
        start_step = mutation_idx * time_interval
        end_step = (mutation_idx + 1) * time_interval
        
        current_abundance = abundance_history[end_step - 1 - 1]
        
        active_species_indices = np.where(current_abundance > threshold)[0]
        
        for species_idx in active_species_indices:
            trait_vector = trait_list[species_idx]
            
            binary_index = int("".join(map(str, trait_vector.astype(int))), 2)
            
            result[mutation_idx, binary_index] = trait_vector
    
    return result


def getSimTraj(conf, traj_num=None, test=False):
    try:
        if test:
            trajectory = np.load(conf.data_dir + 'trajectory_test.npy')
        else:
            trajectory = np.load(conf.data_dir + 'trajectory.npy')
        return trajectory
    
    except:
        os.makedirs(conf.data_dir, exist_ok=True)
        num_resources, total_time, dt, max_mutations = conf.system.num_resources, conf.system.total_time, conf.system.dt, conf.system.max_mutations
        traj_num = conf.system.num_traj if traj_num is None else traj_num
        
        init_num = 1
        initial_abundance = np.array([1.])

        from utilis import random_matrix
        trait_inter_matrix = random_matrix(
            (num_resources, num_resources),
            'tikhonov_sigmoid',
            args={'J_0': 0.5, 'n_star': 4, 'delta': 1},
            triangular=True,
            diagonal=0,
            seed=100
        )
        
        if not test:
            from utilis import plot_trait_interactions
            plot_trait_interactions(trait_inter_matrix, save_path=conf.data_dir)
            
        trajectory = []
        for seed in tqdm(range(1, traj_num+1)):
            np.random.seed(seed)
            init_traits = np.random.randint(0, 2, (init_num, num_resources))
            
            simulator = EcoSimulator(
                n_traits=init_traits,
                num_resources=num_resources,
                trait_inter_matrix=trait_inter_matrix,
            )

            abundance_history = simulator.run(
                x0=initial_abundance,
                total_t=total_time,
                dt=dt,
                mutation_num=max_mutations,
                is_print=True
            )
            
            if seed == 1 and not test:
                from utilis import plot_relative_abundance_dynamics
                plot_relative_abundance_dynamics(abundance_history, save_path=conf.data_dir)
            
            traj = process_abundance_history(abundance_history, simulator.n_traits, num_resources, max_mutations, int(total_time//dt))
            trajectory.append(traj)
        
        if test:
            np.save(conf.data_dir + 'trajectory_test.npy', trajectory)
        else:
            np.save(conf.data_dir + 'trajectory.npy', trajectory)
        
        return np.array(trajectory)
