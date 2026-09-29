# Object-Centric World Model for Pong

## Overview

This project builds a world model for the Atari game Pong, using the object-centric environments from [JAXAtari](https://github.com/k4ntz/JAXAtari). Instead of predicting whole frames, the model predicts the **next state of each object** in the game on its own:

- the **player** paddle
- the **enemy** paddle
- the **ball**

## Main goal: robustness to interventions

The main goal is a world model that stays accurate when the way objects move is **changed at inference time**, so that their movement differs from what was seen during training. For example, the enemy paddle might move with a different behavior, or one object's dynamics might be changed while the others stay the same.

A model that learns one tangled prediction for the whole scene tends to break under such changes, because a change to one object affects its predictions for every object. We want a change to one object's dynamics to affect only the predictions that really depend on that object.

## Approach: independent per-object prediction

To get there, the model predicts **each object's next state independently**. Every object has its own prediction, and that prediction uses only the information it actually needs.

### Dependencies are decided at every step

Which other objects an object depends on is **not fixed**. It is decided again at every time step, based on the current situation.

Pong gives a natural example:

- **Ball near a paddle:** the ball's next position depends on that paddle, since the paddle decides whether and how the ball bounces. The ball's prediction has to take the paddle into account.
- **Ball far from any paddle:** the ball moves on its own. Its next position can be predicted without looking at either paddle.

So the ball depends on a paddle only in the few steps where they interact, and not the rest of the time. The same idea applies to every object.

### Use of history

To predict its next state, each object can use:

- **its own history**, meaning its past states, and
- **the history of the objects it attends to** at the current step.

Information from objects that are not relevant at a given step is left out.

## Visualization of dependencies

We want to be able to **see which other objects each object's prediction needs at every time step**. For any object, and any step of a rollout, it should be clear which objects it depended on, and which it did not.

This makes the model's reasoning something we can check. In Pong, we would expect the ball's prediction to depend on a paddle only in the steps around a bounce, and on no other object while it moves freely. Seeing these dependencies over time lets us:

- check that the dependencies the model picks match what we expect from the game,
- understand how the model behaves when object dynamics are changed, and
- spot cases where the model relies on objects it should not need.

## Robustness and success criterion

We test robustness by changing how an object moves at inference time, using the game modifications that come with JAXAtari. The main test is the **`lazy_enemy`** mod. In it, the enemy paddle only follows the ball while the ball is moving toward the enemy, and stays still otherwise. This is a change to one object's dynamics that the model never sees during training.

**The project goal is met when the actor reaches a final score of at least +10 in Pong with the `lazy_enemy` mod active**, i.e. its points minus the enemy's points at the end of a game (for example 21 : 11). The actor has to win the game for that.

## Why this should help

Because each object's prediction depends only on what matters at that moment, a change to one object's dynamics should stay contained. It should affect only the predictions, and only at the steps, where that object is actually relevant. Everything else should keep working as it did in training. This is what should make the world model robust to changes in how objects move.
