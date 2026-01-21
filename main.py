import random
from dataclasses import dataclass

import torch as t
from tqdm import tqdm

import wandb
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
    name: str = "ttt reinforce"
    random_seed: int = 42
    learning_rate: float = 0.001
    num_episodes: int = 5_000
    gamma: float = 1.0
    rollouts_per_episode: int = 20
    betas: tuple[float, float] = (0.9, 0.99)
    entropy_factor: float = 0.05
    use_baseline: bool = False
    log_interval: int = 100


def main():
    # device = t.device(
    #     "mps"
    #     if t.backends.mps.is_available()
    #     else ("cuda" if t.cuda.is_available() else "cpu")
    # )
    device = t.device("cpu")

    print(f"Using device: {device}")
    config = Config()
    print(f"Config:\n{config}")
    random.seed(config.random_seed)
    t.manual_seed(config.random_seed)

    wandb.init(
        project="rl-from-scratch",
        group="tic-tac-toe",
        name=config.name,
        config=config.__dict__,
    )

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
    vs_random_log: list[tuple[int, ttt.H2HResult]] = []
    for episode in tqdm(range(config.num_episodes)):
        optimizer.zero_grad()
        trajs: list[ttt.Trajectory] = []
        entropy_sum = 0.0
        x_score = 0.0
        o_score = 0.0
        for _rollout_idx in range(config.rollouts_per_episode):
            trajectory = rollout(policy)
            trajs.append(trajectory)
            if config.use_baseline:
                traj_reward = trajectory[-1][2]
                x_score += traj_reward["X"]
                o_score += traj_reward["O"]
            if episode % config.log_interval == 0:
                entropy_sum += ttt.avg_entropy(trajectory)

        if episode % config.log_interval == 0:
            results_vs_random = ttt.results_policy_vs_random(policy)
            vs_random_log.append((episode, results_vs_random))

        x_baseline = x_score / config.rollouts_per_episode
        o_baseline = o_score / config.rollouts_per_episode
        surs = [
            ttt.trajectory_surrogate(
                traj,
                config.gamma,
                config.entropy_factor,
                x_baseline,
                o_baseline,
            )
            for traj in trajs
        ]
        sur = t.stack(surs).mean()
        sur.backward()
        optimizer.step()
        if episode % config.log_interval == 0:
            avg_entropy = entropy_sum / config.rollouts_per_episode
            print(
                f"Episode {episode}: Avg Entropy={avg_entropy:.4f}, Vs Random={vs_random_log[-1][1]}, x_baseline={x_baseline:.4f}, o_baseline={o_baseline:.4f}"
            )
            wandb.log(
                {
                    "avg_entropy": avg_entropy,
                    "x_baseline": x_baseline,
                    "o_baseline": o_baseline,
                    "score_vs_random": vs_random_log[-1][1].score,
                    "x_loss_rate_vs_random": vs_random_log[-1][1].x_loss_rate,
                    "o_loss_rate_vs_random": vs_random_log[-1][1].o_loss_rate,
                },
                step=episode,
            )

    print("Final head-to-head between learned policy and random policy:")
    ttt.h2h(
        policy,
        random_policy,
        policy1_name="Final Learned Policy",
        policy2_name="Random Policy",
    )
    wandb.finish()


if __name__ == "__main__":
    main()
