from typing import List, Optional
from mythologizer.population import AgentAttributeMatrix
from mythologizer.population import EventLog
from mythologizer.population import CultureRegister

class Sandbox:

    def __init__(self, file_path: string, seed: Optional[int] = None, event_log: Optional[EventLog] = EventLog(), population: Optional[AgentAttributeMatrix] = AgentAttributeMatrix(), cultures: Optional[CultureRegister] = None, epoch: int = 0)
        self.file_path = file_path
        self.seed = seed # Maybe refactor this into singleton or pointer
        self.event_log = event_log
        self.population = population
        self.culture_register = cultures
        self.epoch = epoch


    def save_sandbox(self):
        # TODO save sandbox ander serialize all components
        # TODO serialization everywhere
        # binarization
        pass


    def run_epoch(self):
        print(f"running epoch {self.epoch}")
        # update events ("maybe let a new event happen") TODO
        event_log.get_random_new_events()

        # get events in this epoch TODO
        events_in_this_epoch = this.event_log.get_current(this.epoch)

        # get only cultures that changed TODO
        affected_cultures = self.culture_register.affected_by_current_events(events_in_this_epoch)

        #update population on changed cultures. keep in mind that an agent may decide to leave a culture? TODO
        self.population = self.update_to_affected_culture(affected_cultures)

        # let them live TODO
        self.population.run_epoch()

        # save current epoch to file
        self.save_sandbox()
        self.epoch += 1

    def simulate(self, n_epochs = None):
        # maybe add keyboard pause, interrupt etc......
        if n_epochs:
            for _ in range(n_epochs):
                self.run_epoch()
        else:
            while True:
                self.run_epoch()

    def randomly_populate(self, n):
        self.population.random_population(n,self.culture_register, seed)






