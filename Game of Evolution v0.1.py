"""
Game of Evolution v0.1
Chaotic Particle Behavior with Natural Selection

Game of Evolution is an algorithm with which complex structures can emerge by natural selection.
The core of the algorithm is a particle that processes information from its geometrical environment.
The particles react to the presented information with their own unique behavior by spending energy
to influence their geometrical environment. Energy is a limited resource that represents the distance
between particles. Using energy, particles can spawn duplicate particles with mutated behavior,
connect to new particles or elongate existing links. The fragile behaviors get sorted out
automatically because of unsustainable energy spending. Randomness strengthens antifragile behavior.
Many particles together form a decentralized information processing network where every particle is
interacting with each other.
"""

# Imports
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.colors import ListedColormap
import logging
import networkx as nx
matplotlib.use('Agg')

# Total amount of energy inside of simulation
TOTAL_ENERGY = 1000

# Absorption
# Options: RANDOM, MIN_LINKS, MAX_LINKS, MAX_TOT_LINK_ENERGY, MAX_LINK_ENERGY, MAX_AVG_LINK_AVERAGE
ABSORPTION_SURVIVING_CRITERIA = "RANDOM"

# Chance of mutating a value of the weights and biases of the neural network behavior.
MUTATION_CHANCE = 1.0

# Layer information of forward neural network of behaviors.
INPUT_AMOUNT = 15
OUTPUT_AMOUNT = 3
LAYER_SIZES = [9]

#
INPUT_ACTIVATION = np.array([
                            [1],    # [energy],

                            [1],    # [self.node1.particle.link_amount],
                            [1],    # [self.node1.particle.avg_link_energy],
                            [1],    # [self.node1.particle.max_link_energy],
                            [1],    # [self.node1.particle.min_link_energy],
                            [1],    # [self.node1.particle.avg_particle_links],
                            [1],    # [self.node1.particle.max_particle_links],
                            [1],    # [self.node1.particle.min_particle_links],

                            [1],    # [self.node2.particle.link_amount],
                            [1],    # [self.node2.particle.avg_link_energy],
                            [1],    # [self.node2.particle.max_link_energy],
                            [1],    # [self.node2.particle.min_link_energy],
                            [1],    # [self.node2.particle.avg_particle_links],
                            [1],    # [self.node2.particle.max_particle_links],
                            [1],    # [self.node2.particle.min_particle_links],
                            ])

# Plotting options
SMOOTH_PLOTS = 3
PLOT_ITERATIONS = 50
PLOT_EVERY_X_PARTICLE_MOVEMENT = 500


class Behavior:

    """
    Parametric behavior of a particle. Acts as genes of a particle.
    In this case: feed forward neural network
    Input:      geometrical information as input array
    Parameters: weights and biases
    Output:     3 Probabilities for each available action
    """

    weights: [np.array] = None
    biases: [np.array] = None

    def __init__(self, parent_behavior=None):

        self.weights = []
        self.biases = []

        # Random weights and biases
        if parent_behavior is None:
            sizes = [INPUT_AMOUNT] + LAYER_SIZES + [OUTPUT_AMOUNT]
            for i in range(len(sizes) - 1):
                columns = int(sizes[i])
                rows = int(sizes[i + 1])
                self.weights.append(np.random.normal(0.0, 1.0, (rows, columns)))
                self.biases.append(np.random.normal(0.0, 1.0, (rows, 1)))
        # Inheritance from parent behavior
        else:
            for i in range(len(parent_behavior.weights)):
                self.weights.append(parent_behavior.weights[i].copy())
                self.biases.append(parent_behavior.biases[i].copy())

    def get_output(self, input_array: np.array):
        # input must have shape input_amount x 1
        output = input_array
        for i in range(len(self.weights)):
            output = np.maximum(np.matmul(self.weights[i], output) + self.biases[i], 0)
        return output

    def mutate(self):
        # Mutate Values of weights and biases with change MUTATION_CHANCE
        for i in range(len(self.weights)):
            self.weights[i] += np.random.normal(0, 1, self.weights[i].shape) * (np.random.uniform(0, 1, self.weights[i].shape) < MUTATION_CHANCE)
            self.biases[i] += np.random.normal(0, 1, self.biases[i].shape) * (np.random.uniform(0, 1, self.biases[i].shape) < MUTATION_CHANCE)


class LinkNode:

    """
    The intermediary between link and particle
    """

    # Connection references
    link = None
    other_node = None
    particle = None

    input_array: np.array = None

    def initialize(self, particle, link, other_node):
        self.link = link
        self.other_node = other_node
        self.connect(particle)

    def disconnect(self):
        self.particle.nodes.remove(self)
        self.particle = None

    def connect(self, particle):
        self.particle = particle
        self.particle.nodes.append(self)

    def switch_to_particle(self, particle):
        self.disconnect()
        self.connect(particle=particle)

    def create_input_array(self):
        input_array = np.array([
                                [self.link.energy],

                                [self.particle.link_amount],
                                [self.particle.avg_link_energy],
                                [self.particle.max_link_energy],
                                [self.particle.min_link_energy],
                                [self.particle.avg_particle_links],
                                [self.particle.max_particle_links],
                                [self.particle.min_particle_links],

                                [self.other_node.particle.link_amount],
                                [self.other_node.particle.avg_link_energy],
                                [self.other_node.particle.max_link_energy],
                                [self.other_node.particle.min_link_energy],
                                [self.other_node.particle.avg_particle_links],
                                [self.other_node.particle.max_particle_links],
                                [self.other_node.particle.min_particle_links]
                                ])
        self.input_array = input_array * INPUT_ACTIVATION


class Link:

    """
    Links together two particles and holds energy
    """

    node1: LinkNode = None
    node2: LinkNode = None

    energy: int = None

    def __init__(self, particle1, particle2, energy=0):
        self.node1 = LinkNode()
        self.node2 = LinkNode()
        self.node1.initialize(particle=particle1, link=self, other_node=self.node2)
        self.node2.initialize(particle=particle2, link=self, other_node=self.node1)

        self.energy = energy

    def try_absorption(self):
        if self.energy <= 0:
            particles_to_combine = [self.node1.particle, self.node2.particle]
            # Get surviving particle with criteria
            if ABSORPTION_SURVIVING_CRITERIA == "RANDOM":
                surviving_particle = np.random.choice(particles_to_combine)
            elif ABSORPTION_SURVIVING_CRITERIA == "MIN_LINKS":
                surviving_particle = min(particles_to_combine, key=lambda x: x.link_amount)
            elif ABSORPTION_SURVIVING_CRITERIA == "MAX_LINKS":
                surviving_particle = max(particles_to_combine, key=lambda x: x.link_amount)
            elif ABSORPTION_SURVIVING_CRITERIA == "MAX_TOT_LINK_ENERGY":
                surviving_particle = max(particles_to_combine, key=lambda cur_par: cur_par.tot_link_energy)
            elif ABSORPTION_SURVIVING_CRITERIA == "MAX_LINK_ENERGY":
                surviving_particle = max(particles_to_combine, key=lambda cur_par: max(cur_node.link.energy for cur_node in cur_par.nodes))
            elif ABSORPTION_SURVIVING_CRITERIA == "MAX_AVG_LINK_AVERAGE":
                surviving_particle = max(particles_to_combine, key=lambda x: x.avg_link_energy)
            else:
                surviving_particle = None
                logging.warning("NO SELECTION CRITERION CHOSEN")

            particles_to_combine.remove(surviving_particle)
            surviving_particle.absorb_these_particles(particles_to_absorb=particles_to_combine)


class Particle:

    nodes: [LinkNode] = None
    behavior: Behavior = None
    energy: float = None

    simulation = None

    activityID = None
    selected_particle = None
    selected_own_node = None

    link_amount = None
    tot_link_energy = None
    avg_link_energy = None
    max_link_energy = None
    min_link_energy = None
    avg_particle_links = None
    max_particle_links = None
    min_particle_links = None

    def __init__(self, behavior: Behavior, current_simulation):
        self.energy = 0
        self.nodes = []
        self.simulation = current_simulation
        self.behavior = behavior

    def absorb_these_particles(self, particles_to_absorb: []):

        # Step1: connect all links to surviving (self) particle and absorb particles
        for cur_par_to_abs in particles_to_absorb:
            for cur_node in cur_par_to_abs.nodes.copy():
                cur_node.switch_to_particle(particle=self)
            self.energy += cur_par_to_abs.energy
            cur_par_to_abs.energy = 0
            self.simulation.particles.remove(cur_par_to_abs)

        # Step 2: Remove links connected to itself
        # Shouldn't be possible anyway
        for cur_node in self.nodes.copy():
            if cur_node.other_node.particle is self:
                self.energy += cur_node.link.energy
                cur_node.link.energy = 0
                cur_node.other_node.disconnect()
                cur_node.disconnect()
                self.simulation.links.remove(cur_node.link)

        # Step 3: Remove overlapping/double connections and merge energy
        all_unique_connected_particles = []
        all_unique_connected_own_nodes = []
        for cur_node in self.nodes:
            if cur_node.other_node.particle not in all_unique_connected_particles:
                all_unique_connected_particles.append(cur_node.other_node.particle)
                all_unique_connected_own_nodes.append(cur_node)
        for cur_node in self.nodes.copy():
            if cur_node not in all_unique_connected_own_nodes:
                index = all_unique_connected_particles.index(cur_node.other_node.particle)
                actual_node = all_unique_connected_own_nodes[index]
                cur_link = cur_node.link
                actual_node.link.energy += cur_link.energy
                cur_link.energy = 0
                cur_link.node1.disconnect()
                cur_link.node2.disconnect()
                self.simulation.links.remove(cur_link)

    def action_decision(self):

        # If particle has no links, reproduction is only option
        # Should only happen when there is one particle
        if len(self.nodes) == 0 and self.energy > 0:
            self.create_new_particle()
            return

        # Does particle have one energy
        # Should always have
        if self.energy <= 0:
            return

        # Process input arrays, stack outputted probabilities and choose action accordingly
        all_outputs = []
        for curNode in self.nodes:
            output = self.behavior.get_output(input_array=curNode.input_array)
            all_outputs.append(output)
        probabilities = np.hstack(all_outputs)
        shape = probabilities.shape
        amount = shape[0] * shape[1]
        probabilities = probabilities.reshape((amount,))
        if probabilities.sum() <= 0.0:
            p = np.ones(amount) / amount
        else:
            p = probabilities / probabilities.sum()
        pick = np.int64(np.random.choice(np.linspace(0, amount - 1, amount), p=p))
        self.activityID, nodeIDMax = np.unravel_index(pick, shape)
        self.selected_own_node = self.nodes[nodeIDMax]
        self.selected_particle = self.selected_own_node.other_node.particle

        if self.activityID == 0:
            # Action A: Elongate link by one energy
            self.energy -= 1
            self.selected_own_node.link.energy += 1

        elif self.activityID == 1:
            # Action B: Create new link
            self.create_new_link()

        elif self.activityID == 2:
            # Action C: Create new mutated particle
            self.create_new_particle()

    def prepare(self):

        # Prepare geometric information for the input_array
        if len(self.nodes) > 0:
            self.link_amount = len(self.nodes)
            all_link_energy = [cur_node.link.energy for cur_node in self.nodes]
            all_particle_links = [len(curNode.other_node.particle.nodes) for curNode in self.nodes]
            self.tot_link_energy = sum(all_link_energy)
            self.avg_link_energy = self.tot_link_energy / self.link_amount
            self.max_link_energy = max(all_link_energy)
            self.min_link_energy = min(all_link_energy)
            self.avg_particle_links = sum(all_particle_links) / self.link_amount
            self.max_particle_links = max(all_particle_links)
            self.min_particle_links = min(all_particle_links)
        else:
            self.link_amount = 0
            self.tot_link_energy = 0
            self.avg_link_energy = 0
            self.max_link_energy = 0
            self.min_link_energy = 0
            self.avg_particle_links = 0
            self.max_particle_links = 0
            self.min_particle_links = 0

    def create_new_link(self):

        # Link to random neighboring particle of selected particle
        particle_to_link = np.random.choice(self.selected_particle.nodes).other_node.particle

        # Case 1: Selected self, give energy to link
        if self is particle_to_link:
            self.selected_own_node.link.energy += 1
            self.energy -= 1
            return

        # Case 2: Already connected, give energy to according link
        for cur_node in self.nodes:
            if cur_node.other_node.particle is particle_to_link:
                cur_node.link.energy += 1
                self.energy -= 1
                return

        # Case 3: Create new link
        new_link = Link(particle1=self, particle2=particle_to_link)
        new_link.energy += 1
        self.energy -= 1
        self.simulation.links.append(new_link)

    def create_new_particle(self):

        new_gene = Behavior(parent_behavior=self.behavior)

        new_gene.mutate()
        if new_gene is self.behavior:
            logging.warning("Gene is the same which should not be")
        new_particle = Particle(behavior=new_gene, current_simulation=self.simulation)
        new_link = Link(particle1=self, particle2=new_particle)

        # Plotting position (only for visualization)
        if self.simulation.pos is not None:
            self.simulation.pos[new_particle] = self.simulation.pos[self].copy() * np.random.normal(1, 0.001)

        new_link.energy += 1
        self.energy -= 1

        for curNode in self.nodes:
            if curNode is not self.selected_own_node and curNode is not new_link.node1:
                if np.random.uniform(0, 1) > 0.5:
                    curNode.switch_to_particle(new_particle)

        self.simulation.links.append(new_link)
        self.simulation.particles.append(new_particle)


class Simulation:

    iterations_until_next_plot: int = 0
    current_plot: int = 0
    current_iteration = None
    particles: [Particle] = None
    links: [Link] = None
    energy_emitted: int = None

    # For plotting
    pos = None

    def __init__(self):

        self.particles = []
        self.links = []

        self.energy_emitted = 0
        self.current_iteration = 0

        first_particle = Particle(behavior=Behavior(), current_simulation=self)
        self.particles.append(first_particle)

    def run(self, particle_movements=1e15):

        self.iterations_until_next_plot = PLOT_EVERY_X_PARTICLE_MOVEMENT + 1

        while particle_movements > 0:

            # Step 1: choose random particle
            selected_link = None
            selected_particle = np.random.choice(self.particles)

            # Step 2: Get energy
            # Free energy emitting at beginning
            if selected_particle.energy == 0:
                if self.energy_emitted < TOTAL_ENERGY:
                    selected_particle.energy += 1
                    self.energy_emitted += 1
                # Get one energy from random connected link
                else:
                    selected_link = np.random.choice(selected_particle.nodes).link
                    selected_particle.energy += 1
                    selected_link.energy -= 1

            # Step 3: create input arrays
            selected_particle.prepare()
            for curNode in selected_particle.nodes:
                curNode.other_node.particle.prepare()
            for curNode in selected_particle.nodes:
                curNode.create_input_array()

            # Step 4 & 5: Generate action probabilities and choose action accordingly
            selected_particle.action_decision()

            # Step 6: Execute absorption if energy of link is zero
            if selected_link is not None:
                selected_link.try_absorption()

            # Plotting
            self.plot_network()

            self.current_iteration += 1
            particle_movements -= 1

    def plot_network(self):

        """
        Saves images into the Output folder
        """

        self.iterations_until_next_plot -= 1

        if self.iterations_until_next_plot <= 0:
            self.iterations_until_next_plot = PLOT_EVERY_X_PARTICLE_MOVEMENT
        else:
            return

        # Some Error Checks
        if sum([1 for curLink in self.links if curLink.energy < 0]) > 0:
            logging.warning("some links have negative energy")
        if sum([1 for curPar in self.particles if curPar.energy < 0]) > 0:
            logging.warning("some particles have negative energy")
        if sum([1 for curLink in self.links if curLink.energy == 0]) > 0:
            logging.warning("some links have zero energy")
        print(f"#E = {sum([p.energy for p in self.particles]) + sum([l.energy for l in self.links])},"
              f" #P = {len(self.particles)}, #L = {len(self.links)}, #i = {self.current_iteration},"
              f" {ABSORPTION_SURVIVING_CRITERIA}")

        g = nx.Graph()

        for cur_par in self.particles:
            g.add_node(cur_par)
        for cur_link in self.links:
            g.add_edge(cur_link.node1.particle, cur_link.node2.particle, weight=1/(max(cur_link.energy, 1)))

        if self.pos is None:
            self.pos = nx.fruchterman_reingold_layout(g, pos=self.pos, scale=1, iterations=50, dim=3)

        new_pos = nx.fruchterman_reingold_layout(g, pos=self.pos, scale=1, iterations=PLOT_ITERATIONS, dim=3)

        differences = {}
        for part, coord in new_pos.items():
            differences[part] = coord - self.pos[part]

        for i in range(SMOOTH_PLOTS):
            projected_pos = {}
            depth = {}
            for part, diff in differences.items():

                new_coord = self.pos[part] + diff * (1+i) / SMOOTH_PLOTS
                projected_pos[part] = new_coord[0:2]
                depth[part] = new_coord[-1]

            lim = 1.0
            edge_depth = []
            edge_sizes = []
            max_size = 0.9
            min_size = 0.5
            for edge in g.edges:
                cur_depth = (depth[edge[0]]+depth[edge[1]])/2
                cur_depth = min(max(cur_depth, -lim), lim)
                cur_size = min_size + (max_size-min_size)*(cur_depth+lim)/(2*lim)
                edge_depth.append(cur_depth)
                edge_sizes.append(cur_size)

            node_depth = []
            node_sizes = []
            max_size = 25
            min_size = 15
            for curPar in self.particles:
                cur_depth = depth[curPar]
                cur_depth = min(max(cur_depth, -lim), lim)
                cur_size = min_size + (max_size-min_size)*(cur_depth+lim)/(2*lim)
                node_depth.append(cur_depth)
                node_sizes.append(cur_size)

            # binary, BuPu, coolwarm, bwr, cool, YlOrRd, spring
            plt.figure(figsize=(16, 9), dpi=160)
            plt.style.use('dark_background')
            cmap1 = plt.cm.cool
            cmap2 = plt.cm.cool
            cmap1 = cmap1(np.arange(cmap1.N))
            cmap2 = cmap2(np.arange(cmap2.N))

            my_cmap = ListedColormap((cmap1 + cmap2)/2)

            ax = plt.axes()
            options = {
                'ax': ax,
                # 'edge_color': 'black',
                # 'node_color': 'black',
                'node_shape': ".",
                'node_size': node_sizes,    # Array
                'width': edge_sizes,        # Array
                'node_color': node_depth,   # Array
                'edge_color': edge_depth,   # Array

                'cmap': my_cmap,
                'vmin': -lim,
                'vmax':  lim,

                'edge_cmap': my_cmap,
                'edge_vmin': -lim,
                'edge_vmax':  lim,
                'linewidths': 0.0
            }
            plt.subplots_adjust(left=0.2, right=0.8, top=1, bottom=0)
            nx.draw(G=g, pos=projected_pos, with_labels=False, **options)
            ax.margins(x=0.1)

            plt.xlim(-lim, lim)
            plt.ylim(-lim, lim)

            plt.savefig(f"Output\Plot {int(self.current_plot)}, i = {self.current_iteration}.png")
            self.current_plot += 1
            plt.close()

        self.pos = new_pos


if __name__ == "__main__":
    simulation = Simulation()
    simulation.run()
