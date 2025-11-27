to test SPARTA on e.g. the handcrafted blueprint, run something like

conda run -n hanabi python -m experiments.search_full --episodes 3 --seed 0 --rollouts 8 --epsilon 0.1