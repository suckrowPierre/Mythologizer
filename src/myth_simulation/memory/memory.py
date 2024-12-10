from typing import List, Optional
from src.myth_simulation.myths.myth import Myth

class Memory:
    def __init__(self, size):
        self.size = size
        self.memory = List[Myth]

    def remember_myth(self):

    def storing_myth(self, myth: Myth):
        #measure similarity to all other myths
        #take the most similar myth
        #cases:

        # if similarity is 1 - 0.9: do nothing to myth, increase retention
        # get similarity of mythemes and written out myth
        # if similarity is 0.9 - 0.3: combine myth
            # match the mythemes of both
            # if mythemes are not same length or different
            # retention * similarity = delta of change of myth



        # if similarity is 0.3 - 0: store myth in memory + maybe modulation



    def reorder_myth(self):
        ##watch out for retention for new myth if all the others retention are above 1
        ##store retention above 1

    def change_memory_size(self, new_size):
        # reorder arcoring tp

