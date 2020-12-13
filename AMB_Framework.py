# @article{wang2018deep,
#   title={Deep Reinforcement Learning of Cell Movement in the Early Stage of C. elegans Embryogenesis},
#   author={Wang, Zi and Wang, Dali and Li, Chengcheng and Xu, Yichi and Li, Husheng and Bao, Zhirong},
#   journal={arXiv preprint arXiv:1801.04600},
#   year={2018}
# }

from mesa import Agent
import numpy as np

path = "./nuclei/t%03d-nuclei"
cell_list = []
schedule_agents = []
# Cell Agent Class - The Agent in Deep RL is the cell itself
class Cell_Agent(Agent):
    def __init__(self, id, mesa_model, cell_name, cell_location, cell_diameter):
        super().__init__(id, mesa_model)
        self.cell_name = cell_name
        self.cell_location = cell_location
        self.cell_diameter = cell_diameter
        self.state_set()

# Sets the location of the cell based on the name
    def state_set(self):
        if self.name == self.model.cell_type:
            self.model.state_dict[self.name] = self.cell_location
        elif self.name in self.model.cell_state_arr:
            self.model.state_dict[self.name] = self.cell_location

# Moving the cell to the next location
    def move_cell(self):
        self.cell_location = self.forward_location
        self.forward_location = None

# function that combines the move and the state_set of the AMB model
    def step(self):
        self.move_cell()
        self.state_set()

## open the files with the cell features
def ABM_init():
    with open(path) as file:
        for l in file:
            l = l[:len(l) - 1]
            vector = l.split(', ')
            id = int(vector[0])
            # the cell locations are 3D co-ordinates
            cell_location = np.array((vector[5], vector[6], [7]))
            cell_diameter = vector[8]
            name = vector[9]
            if name != '':
                cell_list.append(name)

                # creating a cell_agent for every cell type
                a = Cell_Agent(id, name, cell_location, cell_diameter)

                # adding the cell agents to the scheduler for future use
                schedule_agents.add(a)