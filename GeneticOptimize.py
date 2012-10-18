import numpy as np
import matplotlib.pylab as plt


def runs(bits):
    # make sure all runs of ones are well-bounded
    bounded = np.hstack(([0], bits, [0]))
    # get 1 at run starts and -1 at run ends
    difs = np.diff(bounded)
    run_starts, = np.where(difs > 0)
    return run_starts

def mutate_string(individual):
    ipos = np.random.randint(0,len(individual))
    randPar = parameters()
    randPar = [np.array(randPar[ipos])]
    individual = (individual[0:ipos],randPar,individual[(ipos+1):])
    individual = np.concatenate(individual)
    return individual

def string_crossover(p1,p2):
    ipos = np.random.randint(0,len(p1))
    string = (p1[0:ipos],p2[ipos:])
    string = np.concatenate(string)
    return string


def parameters():
    par =  np.array([0.1*np.random.random_integers(1,100),0.01*np.random.random_integers(1,500),
           np.random.random_integers(-55,-38),np.random.random_integers(-55,-38),
           5*np.random.random_sample(),np.random.random_integers(50,400),
           np.random.random_integers(1,300)])
    return par

def spike_reward_value(Model_mV):
    Model_Spikes = runs(Model_mV > 0)
    spike_reward = 0
    max_spike = len(Model_Spikes)
    if max_spike >= 1:

        interval = 3./INTSTEP
        counter = 0
        spike_test = 0
    
        while spike_test < max_spike and counter < max_spike_data:
            if ( Model_Spikes[spike_test] >= (Data_Spikes[counter] - interval) and
                 Model_Spikes[spike_test] <= (Data_Spikes[counter] + interval) ):
                spike_reward += 1
                counter += 1
                spike_test += 1
            elif Model_Spikes[spike_test] < Data_Spikes[counter] - interval:
                spike_test += 1
            elif Model_Spikes[spike_test] > Data_Spikes[counter] + interval:
                counter += 1
                
    return spike_reward, max_spike

def string_fitness(param):
    VOLTAGE = EXPmodel(param)
    if any(np.isnan(VOLTAGE)):
        error = 1e20
        #print 'skiped one'
        return error
    sumsqr = np.sqrt(sum((voltage - VOLTAGE)**2)/len(voltage))
    dVdt_ERR = np.sqrt(sum(((VOLTAGE - VOLTAGE[::-1]) - (voltage - voltage[::-1]))**2)/len(voltage))
    spike_reward,spike_total = spike_reward_value(VOLTAGE)
    spike_punishment = spike_total - spike_reward
    error = sumsqr + dVdt_ERR + 2*spike_punishment - 5*spike_reward
    return error

def genetic_optimize(population,fitness_function,mutation_function,   
    mate_function, mutation_probability, elite, maxiterations):
# How many winners from each generation?
    original_population_size=len(population)
    top_elite=int(elite*original_population_size)
    # Main loop
    for i in range(maxiterations):
        individual_scores=[(fitness_function(v),v) for v in population]
        individual_scores.sort(key=lambda v:v[0])
        ranked_individuals=[v for (s,v) in individual_scores]
        # Start with the pure winners
        population=ranked_individuals[0:top_elite]
        
    # Add mutated and bred forms of the winners
        while len(population)<original_population_size:
            if np.random.random()<mutation_probability:            
                # Mutation
                c=np.random.randint(0,top_elite)
                population.append(mutation_function(ranked_individuals[c]))
            else:
                # Crossover
                c1= np.random.randint(0,top_elite)
                c2= np.random.randint(0,top_elite)
                if c1 == c2:
                    c2 = c2+1
                population.append(mate_function(ranked_individuals[c1],ranked_individuals[c2]))
    return individual_scores[0][1], i+1
	
	
#### GEN INITIAL POPULATION ####
param_pop = np.zeros((50,7))
for i in range(0,len(param_pop)):
    param_pop[i,:] = parameters()


Data_Spikes = runs(voltage > 0)
max_spike_data = len(Data_Spikes)


EvolvedPopulation, NumberGen = genetic_optimize(param_pop, string_fitness, mutate_string, string_crossover, 0.50,0.2,NUMGEN)

np.savetxt(SAVE_FILENAME,EvolvedPopulation, delimiter=',')

if PLOT1 is 1:
    VOLTAGE = EXPmodel(EvolvedPopulation)
    x_ = np.arange(0,TIME,INTSTEP)
    
    fig = plt.figure()
    ax = fig.add_subplot(211)
    ax.plot(x_, VOLTAGE, 'b', x_, voltage, 'k',linewidth=2)
    plt.legend(('model','data'),loc='upper right')
    plt.yticks(fontsize = 16)
    plt.ylabel('mem. pot. (mV)', fontsize = 16)
    ax3 = fig.add_subplot(212)
    ax3.plot(x_, CURRENT, 'k', linewidth=2)
    plt.yticks(fontsize = 16)
    plt.ylabel('current (pA)', fontsize = 16)
    plt.xlabel('time (msec)', fontsize = 16)
    plt.tight_layout()
    plt.show()