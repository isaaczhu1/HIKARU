Since this is not a software project:

A _blueprint_ simply is a wrapper for an act() function. This structure permits the persistence of information across multiple actions. This behavior is NOT ENFORCED by any abstract class.

This act() function inputs a HanabiObservation and an rng. The rng must be used to deterministically sample from the act(). It furthermore absolutely MUST NOT mutate the rng, e.g. by restoring the random state at the end of the call.

The SPARTA-GRU implementation must be exceedingly careful to properly follow the GRU state, or else it will explode.