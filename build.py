import matplotlib.pyplot as plt
import animatplot as amp
import numpy as np
import random
import time
import copy

class DynamicTuner():
    '''
    This class watches detects if improvements have stalled,
    and tries to dynamically change mutation levels.
    '''

    def __init__(self,model=None,max_epochs=-1):
        assert model, 'You must include a model parameter.'
        self.model = model
        self.mutate_proba = 0.001
        max_epochs = int(max_epochs)
        self.max_epochs = max_epochs

    def run(self):

        while True:
            self.current_epoch = len(self.model.generation_list)
            self.next_target = self.current_epoch + 100
            self.next_target = self.current_epoch + 1

            # Check conditions
            self.early_high_mutate()
            self.early_high_pop()

            if not self.max_epochs == -1:
                if self.max_epochs < self.current_epoch:
                    break

            self.model.max_epochs = self.next_target
            self.model.mutate_proba = self.mutate_proba
            self.model.run()



        return self

    def early_high_mutate(self):
        '''
        Increases mutation rate in earlier epochs
        '''

        self.mutate_proba = np.log2(self.next_target+20000) / (self.next_target+20000)
        self.mutate_proba = self.mutate_proba

        return self

    def early_high_pop(self):
        '''
        Increases population size in earlier epochs. Shock
        the population with different sizes of mating pools.
        '''

        target_map = {  '20':250,
                        '40':225,
                        '60':200,
                        '100':150,
                        '200':100,
                        '300':75,
                        '400':50,
                        '500':60,
                        '600':70,
                        '700':80,
                        '800':150,
                        '900':125,
                        '1000':100,
                        '1100':85,
                        '1200':100,
                        '1300':150,
                        '1400':175,
                        '1500':200
                        }

        last_key = '0'
        for key in target_map:
            if self.next_target > float(last_key) \
            and self.next_target < float(key):
                self.model.N = target_map[key]
            last_key = key

        return self



class GeneticAlgo():
    '''

    Evolves a world over time using penalty and rewards.
    You can run() the function indefinitely. You can also
    interrupt the function, set model.auto_plot=True,
    can continue running with model.run()

    Usage:
    model = GeneticAlgo(N=150,mutate_proba=0.0002,max_epochs=1000,auto_plot=False)
    model.run()

    Expects:
        N INT the population size used for selection
        mutate_proba FLOAT the probability of mutating a child
    '''

    def __init__(
            self,
            N=15,
            mutate_proba=0.01,
            max_epochs=10,
            auto_plot=False):

        # Size of the world. E.g: 10x10
        self.size = (50,50)

        N = int(N)
        mutate_proba = float(mutate_proba)
        self.N = N
        self.mutate_proba = mutate_proba
        self.best_child = None
        self.intitialized = False

        assert isinstance(
            auto_plot, bool), 'Expected auto_plot to be BOOL, got type: {}'.format(
            type(auto_plot))
        self.auto_plot = auto_plot

        assert isinstance(
            max_epochs, int), 'Expected max_epochs to be INT, got type: {}'.format(
            type(max_epochs))
        self.max_epochs = max_epochs



    def initialize(self):
        'Create population of world elements.'

        self.generation_list = []
        self.generation_fitness = []
        self.current_pop = [World(*self.size).gen_world()
                            for i in range(self.N)]
        self.next_pop = []
        self.intitialized = True
        return self

    def selection(self):
        'Evaluate fitness of population and build mating pool from current pop.'

        self.current_pop_fitness = [
            Fitness(w).run().fitness_value for w in self.current_pop]

        # exp func -- Seems to really help polarize best performers!
        exp_func = 3
        self.current_pop_fitness = [
            x**exp_func for x in self.current_pop_fitness]
        return self

    def reproduction(self):
        '''
        Generates and mutates child until the current pop is completed.
        '''

        for i in range(self.N):
            parents = self.pick_parents()
            child = self.crossover(parents)
            child = self.mutation(child)
            self.birth_child(child)

        return self

    def pick_parents(self):
        '''
        Choose parents (probability based on fitness)
        '''

        parents = random.choices(
            population=self.current_pop,
            weights=self.current_pop_fitness,
            k=2)
        return parents

    def crossover(self, parents):
        '''
        Combine DNA of parents.
        '''

        # For the target 2d array size, take random
        # DNA from either of the 2 parents
        child = World(*self.size)
        child.gen_world()

        i = 0
        for y in range(self.size[0]):
            for x in range(self.size[1]):
                parent_selection = random.choice([0, 1])
                child.world[y][x] = parents[parent_selection].world[y][x]
                i += 1

        # Make sure that all data points have been iterated over.
        parent_len = parents[0].world.shape[0] * parents[0].world.shape[1]
        assert i == parent_len, 'All data points were not iterated over. Got: {} Expected: {}'.format(
            i, parent_len)

        #
        # i = 0
        # for y in range(self.size[0]):
        #         parent_selection = random.choice([0,1])
        #         child.world[y] = parents[parent_selection].world[y]
        #         i+=1

        return child

    def mutation(self, child):
        '''
        Slightly mutate the child based on probability of self.mutate_proba
        '''

        i = 0
        for y in range(self.size[0]):
            for x in range(self.size[1]):

                # Only mutate if the self.mutate_proba event is picked.
                chance_to_mutate = random.random() < self.mutate_proba
                if chance_to_mutate == 1:
                    child.world[y][x] = random.choice([0, 1])
                i += 1

        child_len = child.world.shape[0] * child.world.shape[1]
        assert i == child_len, 'All data points were not iterated over. Got: {} Expected: {}'.format(
            i, parent_len)
        return child

    def birth_child(self, child):
        '''
        Add a child to the next population generation.
        '''

        self.next_pop.append(child)
        return self

    def update_pop(self):
        '''
        Replace the current pop with nextGen pop.
        '''

        self.current_pop = self.next_pop
        self.next_pop = []

        return self

    def run(self):

        if not self.intitialized:
            self.initialize()
        while True:

            self.selection()
            self.reproduction()

            # Save self.best_child
            self.best_child = self.current_pop[np.argmax(
                self.current_pop_fitness)]
            self.generation_list.append(self.best_child)

            # Get fitness of best child
            f = Fitness(self.best_child)
            f.run()
            best_child_fitness = round(f.fitness_value, 2)
            self.generation_fitness.append(best_child_fitness)

            if len(self.generation_list) % 20 == 1:
                print('Generation: {} Fitness: {}'.format(
                    len(self.generation_list), best_child_fitness))

            if self.auto_plot:
                self.best_child.init_world()
                self.best_child.show_world()
                plt.show()
                plt.title('Generation: {} Fitness: {}'.format(
                    len(self.generation_list), best_child_fitness))

            # Update to the next generation
            self.update_pop()

            # If the max_epochs is met, then stop.
            if self.max_epochs != -1:
                if len(self.generation_list) > self.max_epochs:
                    break
        return self


class Fitness():
    '''
    Defines an viewer object which is able to move through the world.
    '''

    def __init__(self, world):

        try:
            self.world = copy.deepcopy(world)
        except NotImplementedError:
            self.world = copy.copy(world)

        self.start_pos = (0, 0)
        self.penalty_map = None
        self.fitness_value = 50

        self.movement_condition_map = {
            'traversable': 0,
            'non-traversable': 1,
            'connected_to_start_pos': 2}

        self.penalty_condition_values = {
            'turns': -.018,
            '2fork':-.8,
            '3fork':-2,
            'secondary_paths':-.03,
            'vertigo':-.03,
            'noise':-.018,
            'extra_paths':-.018}

        self.pathfinder_run = False

    def run(self):

        self.init_penalty_map()
        self.pathfinder()
        self.path_too_wide()
        self.fewer_turns()
        self.secondary_paths()
        self.vertigo()
        self.denoise_world()
        self.penalty_for_path()

        # Path penalty
        self.fitness_value += np.sum(self.penalty_map)

        # Turn penalty
        self.fitness_value += self.fewer_turns_penalty * \
            self.penalty_condition_values['turns']

        # Secondary forking paths
        self.fitness_value += self.secondary_paths_penalty * \
            self.penalty_condition_values['secondary_paths']

        # Vertigo index
        self.fitness_value += self.vertigo_index * \
            self.penalty_condition_values['vertigo']

        # Denoise world
        self.fitness_value += self.noise_on_map_penalty * \
            self.penalty_condition_values['noise']

        # Clean world
        self.fitness_value += self.extra_traversable_penalty * \
            self.penalty_condition_values['extra_paths']

        # Add bonuses
        self.fitness_value += (self.depth_max*1.5)

        return self

    def init_penalty_map(self):
        self.penalty_map = np.zeros(
            (self.world.x_bounds[1], self.world.y_bounds[1]))

        assert len(
            self.penalty_map.shape) == len(
            self.world.world.shape), 'self.penalty_map contained a different shape than self.world, {} {}'.format(
            self.penalty_map.shape, self.world)
        assert self.penalty_map.shape[0] == self.world.world.shape[0] and self.penalty_map.shape[1] == self.world.world.shape[
            1], 'self.penalty_map contained a different shape than self.world, {} {}'.format(self.penalty_map.shape, self.world)
        return self

    def vertigo(self):
        '''
        Penalty if right / left turns are imbalanced.
        '''

        right_turns = 0
        left_turns = 0
        for i in range(len(self.pathfinder_step_entries)):
            if i>0:
                if (self.pathfinder_step_entries[i-1] == 'up' and \
                self.pathfinder_step_entries[i] == 'right') or \
                (self.pathfinder_step_entries[i-1] == 'right' and \
                self.pathfinder_step_entries[i] == 'down') or \
                (self.pathfinder_step_entries[i-1] == 'down' and \
                self.pathfinder_step_entries[i] == 'left') or \
                (self.pathfinder_step_entries[i-1] == 'left' and \
                self.pathfinder_step_entries[i] == 'up'):
                    right_turns+=1

            if i+2==len(self.pathfinder_step_entries):
                if (self.pathfinder_step_entries[i] == 'up' and \
                self.pathfinder_step_entries[i-1] == 'right') or \
                (self.pathfinder_step_entries[i] == 'right' and \
                self.pathfinder_step_entries[i-1] == 'down') or \
                (self.pathfinder_step_entries[i] == 'down' and \
                self.pathfinder_step_entries[i-1] == 'left') or \
                (self.pathfinder_step_entries[i] == 'left' and \
                self.pathfinder_step_entries[i-1] == 'up'):
                    left_turns+=1

        self.vertigo_index = abs(left_turns - right_turns)
        return self

    def secondary_paths(self):
        '''
        Penalty if there are too many forking paths. There should only
        be one primary path.
        '''

        self.secondary_paths_penalty = self.pathfinder_steps - self.depth_max
        return self

    def denoise_world(self):
        '''
        Penalty if world is left messy.
        '''

        self.noise_on_map_penalty = 0

        for y in range(self.world.world.shape[0]):
            for x in range(self.world.world.shape[1]):

                num_adjacent = 0
                try:
                    if self.world.world[y][x] == 0 and self.world.world[y+1][x] == 1:
                        num_adjacent += 1
                except IndexError:
                    continue
                try:
                    if self.world.world[y][x] == 0 and self.world.world[y-1][x] == 1:
                        num_adjacent += 1
                except IndexError:
                    continue
                try:
                    if self.world.world[y][x] == 0 and self.world.world[y][x+1] == 1:
                        num_adjacent += 1
                except IndexError:
                    continue
                try:
                    if self.world.world[y][x] == 0 and self.world.world[y][x-1] == 1:
                        num_adjacent += 1
                except IndexError:
                    continue

            if num_adjacent > 2:
                self.noise_on_map_penalty +=1

        return self

    def penalty_for_path(self):
        '''
        Slight penalty for spaces that are traversable. This provides
        incentive to remove unused paths.
        '''

        self.extra_traversable_penalty = 0

        for y in range(self.world.world.shape[0]):
            for x in range(self.world.world.shape[1]):
                if self.world.world[y][x] == 0:
                    self.extra_traversable_penalty +=1
        return self

    def path_too_wide(self):
        '''
        Penalty if there are too many adjacent traversable paths,
        the world space is likely not as efficient as it could be.
        '''

        assert len(self.penalty_map) > 0, 'Must run init_penalty_map() first.'
        assert self.pathfinder_run, 'Must run pathfinder() first.'
        allowed_adjacent_traversable = 2

        for y in range(self.penalty_map.shape[0]):
            for x in range(self.penalty_map.shape[1]):

                num_adjacent = 0

                if self.world.world[y][x] == self.movement_condition_map['connected_to_start_pos']:
                    num_adjacent += 1

                try:
                    if self.world.world[y][x] == self.movement_condition_map['connected_to_start_pos'] \
                            and self.world.world[y + 1][x] == self.movement_condition_map['connected_to_start_pos']:
                        num_adjacent += 1
                except IndexError:
                    continue

                try:
                    if self.world.world[y][x] == self.movement_condition_map['connected_to_start_pos'] \
                            and self.world.world[y - 1][x] == self.movement_condition_map['connected_to_start_pos']:
                        num_adjacent += 1
                except IndexError:
                    continue

                try:
                    if self.world.world[y][x] == self.movement_condition_map['connected_to_start_pos'] \
                            and self.world.world[y][x + 1] == self.movement_condition_map['connected_to_start_pos']:
                        num_adjacent += 1
                except IndexError:
                    continue

                try:
                    if self.world.world[y][x] == self.movement_condition_map['connected_to_start_pos'] \
                            and self.world.world[y][x - 1] == self.movement_condition_map['connected_to_start_pos']:
                        num_adjacent += 1
                except IndexError:
                    continue

                # If the number of traversable adjacent spaces is over
                # the limit, then add penalty.
                if num_adjacent == 4:
                    self.penalty_map[y][x] += \
                        self.penalty_condition_values['2fork']
                if num_adjacent == 5:
                    self.penalty_map[y][x] += \
                        self.penalty_condition_values['3fork']

        return self

    def fewer_turns(self):
        '''
        Penalty if too many turns are being taken.
        '''

        ideal_path_len = 16

        self.fewer_turns_penalty = 0
        for i in range(ideal_path_len):
            roll = np.roll([x for x in self.pathfinder_step_entries], shift=8)
            diff_dir = self.pathfinder_step_entries != roll
            self.fewer_turns_penalty += np.sum(diff_dir)
        return self

    def pathfinder(self):
        '''
        Uses memoized floodfill algorithm to find the longest available path.
        '''

        self.pathfinder_steps = 0
        self.pathfinder_step_entries = []

        def floodfill(matrix, x, y, direction='left',depth=0):
            if matrix[y][x] == 0:

                # Update plot - time consuming!
                try:
                    self.world.ax.plot(
                        x, y, 'o', color='red', zorder=1, linewidth=6)
                    plt.draw()
                except AttributeError:
                    pass


                # # Check if this was a turn and append
                # if self.pathfinder_step_entries[-1]!=direction:
                self.pathfinder_steps += 1
                self.pathfinder_step_entries.append(direction)

                # Update path as 'connected_to_start_pos'
                matrix[y][x] = self.movement_condition_map['connected_to_start_pos']

                # Get max depth of the current path
                depth +=1
                self.depth_array.append(depth)

                # Recursively invoke flood fill on all surrounding pixels:
                if x > 0:
                    floodfill(matrix, x - 1, y, 'left',depth)


                if x < len(matrix[y]) - 1:
                    floodfill(matrix, x + 1, y, 'right',depth)

                if y > 0:
                    floodfill(matrix, x, y - 1, 'down',depth)

                if y < len(matrix) - 1:
                    floodfill(matrix, x, y + 1, 'up',depth)

        matrix = self.world.world
        self.depth_array = [0]
        floodfill(matrix, *self.start_pos)
        self.depth_max = np.amax(self.depth_array)
        self.pathfinder_run = True
        return self

    def render_view_start(self):

        try:
            # Try to plot if an ax is available
            self.world.ax.plot(
                *self.start_pos,
                'o',
                color='red',
                zorder=1,
                linewidth=6)
        except AttributeError:
            pass

        return self


class World():
    '''
    Defines an array of line segments which represent square primatives
    for defining world collision surfaces.
    '''

    def __init__(self, x_lim, y_lim):

        self.hello = 'world'
        assert isinstance(x_lim, int), 'x_lim must be an INT'
        assert isinstance(y_lim, int), 'y_lim must be an INT'
        self.x_bounds = (0, x_lim)
        self.y_bounds = (0, y_lim)
        self.finished_world = False

    def gen_fig(self):
        self.fig = plt.figure(1, figsize=(10, 10), dpi=90)
        self.ax = self.fig.add_subplot(1, 1, 1)

    def init_world(self):
        self.gen_fig()
        return self

    def gen_world(self):
        '''
        Generates a random numpy array
        '''

        self.world = np.random.rand(self.x_bounds[1], self.y_bounds[1]).round()
        return self

    def check_constraints(self):
        '''
        Checks if constraints are true after generating a world
        '''

        return self

    def show_world(self):
        '''
        Returns fig,ax of current best child.
        '''
        self.ax.imshow(self.world)
        return self.fig,self.ax


class WorldGenerator():
    '''
    A class for generating a pixel map based on objectives.

    Expects:
    objective DICT containing keypairs of recognized criteria.

    E.g. usage:
    world = WorldGenerator({'min_path_steps':25})
    world.run()
    '''

    def __init__(self, objective=None, size=None):

        assert objective, 'No objective was set'
        assert isinstance(
            size, tuple), 'Expected size TUPLE. E.g: (10,10), got type: {}.'.format(
            type(size))
        self.size = size
        self.objective = objective
        self.best_world = None
        self.best_fitness = -np.inf
        self.fitness_trial_array = []

    def run(self, objective=None):
        '''
        Run the fitness objective. This can be re-ran at any time
        to continue caching older trials.
        '''

        if objective:
            self.objective = objective
        assert self.objective, 'No objective was set for WorldGenerator()'
        if 'fitness_goal' in self.objective:
            self.fitness_goal(self.objective['fitness_goal'])

        return self

    def fitness_goal(self, fitness_goal=None):

        assert fitness_goal, 'fitness_goal was not a valid value.'
        fitness_goal = float(fitness_goal)
        assert isinstance(fitness_goal, float)

        w = World(*self.size)
        w.init_world()

        # Generate some world
        while True:

            # Generate new world
            w.ax.cla()
            w.gen_world()
            w.show_world()

            f = Fitness(w)
            f.run()
            self.fitness_trial_array.append(f.fitness_value)

            if f.fitness_value > self.best_fitness:
                self.best_world = {'world': w,
                                   'fitness': f}

            # Continue until map is reasonable
            if f.fitness_value > self.objective['fitness_goal']:
                break
        print('Best fitness:', self.best_world['fitness'].fitness_value)
        self.best_world = {'world': w,
                           'fitness': f}

        return self
