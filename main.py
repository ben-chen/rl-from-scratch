import random
from dataclasses import dataclass

import torch as t
from tqdm import tqdm

from environments import tic_tac_toe as ttt


def rollout(policy: ttt.TorchPolicy) -> ttt.Trajectory:
    trajectory: ttt.Trajectory = []
    state = ttt.State(current_player="X")
    while state.check_winner() == "NotFinished":
        probs = policy.probs(state)
        action = policy.choose_action(probs)
        next_state = state.transition(action)
        reward = ttt.reward(next_state)
        trajectory.append((state, action, reward, probs))
        state = next_state

    return trajectory


@dataclass
class Config:
    random_seed: int = 42
    learning_rate: float = 0.001
    num_episodes: int = 20_000
    gamma: float = 1.0
    rollouts_per_episode: int = 20
    betas: tuple[float, float] = (0.9, 0.99)
    entropy_factor: float = 0.05


def main():
    # device = t.device(
    #     "mps"
    #     if t.backends.mps.is_available()
    #     else ("cuda" if t.cuda.is_available() else "cpu")
    # )
    device = t.device("cpu")

    print(f"Using device: {device}")
    config = Config()
    random.seed(config.random_seed)
    t.manual_seed(config.random_seed)

    print("Initializing model and policies...")
    model = ttt.Mlp(device)
    parameters = model.parameters()
    policy = ttt.TorchPolicy(model)
    random_policy = ttt.RandomPolicy()

    print("Initial head-to-head between learned policy and random policy:")
    ttt.h2h(
        policy,
        random_policy,
        policy1_name="Initial Learned Policy",
        policy2_name="Random Policy",
    )

    optimizer = t.optim.AdamW(
        parameters, lr=config.learning_rate, betas=config.betas, maximize=True
    )
    for episode in tqdm(range(config.num_episodes)):
        optimizer.zero_grad()
        surs: list[t.Tensor] = []
        entropy_sum = 0.0
        for _rollout_idx in range(config.rollouts_per_episode):
            trajectory = rollout(policy)
            surs.append(
                ttt.trajectory_surrogate(
                    trajectory, config.gamma, config.entropy_factor
                )
            )
            if episode % 1000 == 0:
                entropy_sum += ttt.avg_entropy(trajectory)
        sur = t.stack(surs).mean()
        sur.backward()
        optimizer.step()
        if episode % 1000 == 0:
            avg_entropy = entropy_sum / config.rollouts_per_episode
            print(f"Episode {episode}: Avg Entropy={avg_entropy:.4f}")

    print("Final head-to-head between learned policy and random policy:")
    ttt.h2h(
        policy,
        random_policy,
        policy1_name="Final Learned Policy",
        policy2_name="Random Policy",
    )


if __name__ == "__main__":
    main()
