import random
from dataclasses import dataclass

import torch as t
from tqdm import tqdm

from environments import tic_tac_toe as ttt


def rollout(policy: ttt.TorchPolicy) -> ttt.Trajectory:
    trajectory: ttt.Trajectory = []
    state = ttt.State(current_player="X")
    while state.check_winner() == "NotFinished":
        logits = policy.probs(state)
        action = policy.choose_action(logits)
        next_state = state.transition(action)
        reward = ttt.reward(next_state)
        trajectory.append((state, action, reward, logits))
        state = next_state

    return trajectory


@dataclass
class Config:
    random_seed: int = 42
    learning_rate: float = 0.01
    num_episodes: int = 5_000
    gamma: float = 1.0
    rollouts_per_episode: int = 50


def main():
    config = Config()
    random.seed(config.random_seed)
    t.manual_seed(config.random_seed)

    model = ttt.Mlp()
    parameters = model.parameters()
    policy = ttt.TorchPolicy(model)
    random_policy = ttt.RandomPolicy()

    ttt.h2h(
        policy,
        random_policy,
        policy1_name="Initial Learned Policy",
        policy2_name="Random Policy",
    )

    optimizer = t.optim.SGD(parameters, lr=config.learning_rate, maximize=True)
    for episode in tqdm(range(config.num_episodes)):
        optimizer.zero_grad()
        surs: list[t.Tensor] = []
        for _rollout_idx in range(config.rollouts_per_episode):
            trajectory = rollout(policy)
            surs.append(ttt.trajectory_surrogate(trajectory, config.gamma))
        sur = t.stack(surs).mean()
        sur.backward()
        optimizer.step()

    ttt.h2h(
        policy,
        random_policy,
        policy1_name="Final Learned Policy",
        policy2_name="Random Policy",
    )


if __name__ == "__main__":
    main()
