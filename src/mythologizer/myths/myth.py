from typing import List, Optional
import uuid


class Myth:

    def __init__(self, myth_written_out, mythemes, original_myth_id):
        self.id = uuid.uuid4()
        self.current_myth = myth_written_out
        self.current_mythemes = List[str]
        self.retention = 1.0
        self.original_myth_Id = original_myth_id

    def __eq__(self, other):
        return self.id == other.id

    def same_old_myth(self, other) -> bool:
        return self.original_myth_Id == other.original_myth_Id

    def compare_mythemes(self, other) -> float:
    ### 0 is no similarity, 1 is identical
    ### use fuzzy matching to compare mythemes
    ### maybe embeddings ? maybe spacy??
        return 0

    def compare_written_out_myth(self, other) -> float:
    ### 0 is no similarity, 1 is identical
    ### maybe embeddings ? maybe spacy??
        return 0

    def modify_retention(self, delta: float):
        self.retention = self.retention + delta
        if self.retention < 0:
            ## TODO: save dying myths somwehere
            del self


    def merge_with_myth(self, other, weight: float, merge_function):

        # weight = weight_larry
        # 1 = weight_bob + weight_larry
        weight_given_myth = (1 - weight)

        updated_myth = merge_function(self, weight, other, weight_given_myth)
        self = updated_myth

        # update mythemes

        #result merged text