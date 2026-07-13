# bcWarmSoftFilterValSelect

## Overview

`bcWarmSoftFilterValSelect` is a behavior-regularized DDPG controller for delayed reheat-steam-temperature control. The public snapshot documents the algorithm and its locked configuration without publishing plant data, trained weights, trajectories, or evaluation artifacts.

## Actor objective

The actor is optimized using

\[
L_{actor}=-Q(s,\pi(s))+\lambda_t L_{BC}.
\]

The first term optimizes the critic-estimated return. The second term regularizes the actor toward selected historical actions. Its coefficient decays during training so that demonstrations exert their strongest influence early in optimization.

## Main components

1. **Decaying behavior regularization.** The behavior-cloning coefficient decreases from 1.0 to 0.05 over 120 episodes.
2. **Critic-based Q-filter.** The behavior-cloning loss is activated when the critic assigns the historical action a higher estimated value than the current policy action.
3. **Heuristic soft weighting.** Flagged demonstrations are downweighted instead of removed, preserving operating-condition coverage.
4. **Validation-based selection.** The actor is evaluated every 10 episodes and selected by development-set MAE.

The heuristic weights are engineering choices. They should not be interpreted as causal demonstration-quality estimates or theoretically optimal confidence scores.

## Public files

- `src/train_controller_ddpg.py`: controller training and selection logic.
- `src/evaluate_controller.py`: evaluation entry point.
- `configs/bcWarmSoftFilterValSelect.env.example`: sanitized configuration template.

## Usage

Copy the example configuration into the process environment, replace both data placeholders with authorized local paths, and run from the repository root:

```text
python src/train_controller_ddpg.py
```

For evaluation, set `ACTOR_PATH_OVERRIDE` to an authorized local actor checkpoint before running:

```text
python src/evaluate_controller.py
```

## Scope

This public snapshot defines a research method. It does not include evidence for field deployment, energy-efficiency improvement, fuel-consumption reduction, emission reduction, or equipment-life extension. Those claims require separate measurements and validation.
