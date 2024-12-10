
from agents import BaseAgent

## Example Bob meets Larry

Bob = BaseAgent("Bob")
Larry = BaseAgent("Larry")

# Larry and Bob are close to each other

# Communciation Handler

communicate(Bob,Larry)
    # Bob speaks to Larry
    myth_bob = Bob.memory.get_with_probability() # not defined
    myth_bob.modulate(interaction, attributes_bob) # not defined
    Larry.memory.storing_myth(myth_bob) # not defined
        myth_larry = Larry.memory.closest_myth(myth_bob)  # not defined
        if myth_larry is None:
            Larry.memory.storing_myth(myth_bob)
        else:
            # calculate weight_larry depending on interaction and attributes of both
            # myth_larry.merge(myth_bob, weight_larry) # not defined
            # Larry.memory.update() # not defined






