from dataclasses import dataclass
from typing import Literal, Self

import torch

type Player = Literal["X", "O"]
type Action = tuple[int, int]  # (row, column)
type Reward = dict[Player, float]
type Prob = torch.Tensor
type Step = tuple[State, Action, Reward, Prob]
type Trajectory = list[Step]


class State:
    def __init__(self, current_player: Player):
        self.board = [[" " for _ in range(3)] for _ in range(3)]
        self.current_player: Player = current_player
        self.other_player: Player = "O" if current_player == "X" else "X"

    def check_winner(self) -> Player | Literal["Draw", "NotFinished"]:
        if (
            any(
                all(self.board[r][c] == self.current_player for c in range(3))
                for r in range(3)
            )
            or any(
                all(self.board[r][c] == self.current_player for r in range(3))
                for c in range(3)
            )
            or all(self.board[i][i] == self.current_player for i in range(3))
            or all(self.board[i][2 - i] == self.current_player for i in range(3))
        ):
            return self.current_player
        elif (
            any(
                all(self.board[r][c] == self.other_player for c in range(3))
                for r in range(3)
            )
            or any(
                all(self.board[r][c] == self.other_player for r in range(3))
                for c in range(3)
            )
            or all(self.board[i][i] == self.other_player for i in range(3))
            or all(self.board[i][2 - i] == self.other_player for i in range(3))
        ):
            return self.other_player
        elif all(self.board[r][c] != " " for r in range(3) for c in range(3)):
            return "Draw"
        else:
            return "NotFinished"

    def valid_moves(self) -> list[Action]:
        if self.check_winner() != "NotFinished":
            return []
        return [(r, c) for r in range(3) for c in range(3) if self.board[r][c] == " "]

    def reward(self) -> Reward:
        winner = self.check_winner()
        if winner == "Draw" or winner == "NotFinished":
            return {"X": 0.0, "O": 0.0}
        elif winner == "X":
            return {"X": 1.0, "O": -1.0}
        else:
            return {"X": -1.0, "O": 1.0}

    def transition(self, action: Action) -> Self:
        r, c = action
        if self.board[r][c] != " ":
            raise ValueError("Invalid move")

        if self.check_winner() != "NotFinished":
            raise ValueError("Game has already ended")

        new_state = type(self)(self.other_player)
        new_state.board = [row[:] for row in self.board]
        new_state.board[r][c] = self.current_player
        return new_state

    def __str__(self) -> str:
        rows = ["|".join(self.board[r]) for r in range(3)]
        return "\n-----\n".join(rows)


def reward(state: State) -> Reward:
    return state.reward()


class Policy:
    def select_action(self, state: State) -> Action:
        raise NotImplementedError


class RandomPolicy(Policy):
    import random

    def select_action(self, state: State) -> Action:
        valid_moves = state.valid_moves()
        return self.random.choice(valid_moves)


def to_flat(action: Action) -> int:
    r, c = action
    return r * 3 + c


def from_flat(index: int) -> Action:
    return (index // 3, index % 3)


class TorchPolicy(Policy):
    def __init__(self, model: torch.nn.Module):
        self.model = model

    @staticmethod
    def state_to_tensor(state: State) -> torch.Tensor:
        if state.current_player == "X":
            x_val, o_val = 1.0, -1.0
        else:
            x_val, o_val = -1.0, 1.0
        mapping = {"X": x_val, "O": o_val, " ": 0.0}
        board_tensor = torch.tensor(
            [[mapping[state.board[r][c]] for c in range(3)] for r in range(3)],
            dtype=torch.float32,
        ).flatten()
        return board_tensor

    def probs(self, state: State) -> torch.Tensor:
        state_tensor = self.state_to_tensor(state)
        logits = self.model(state_tensor)
        valid_moves = state.valid_moves()
        valid_indices = [to_flat(action) for action in valid_moves]
        valid_mask = torch.full((9,), False)
        valid_mask[valid_indices] = True
        logits[~valid_mask] = float("-inf")
        probs = torch.softmax(logits, dim=0)
        return probs

    def choose_action(self, probs: torch.Tensor) -> Action:
        chosen_index = int(torch.multinomial(probs, 1).item())
        chosen_move = from_flat(chosen_index)
        return chosen_move

    def select_action(self, state: State) -> Action:
        probs = self.probs(state)
        chosen_move = self.choose_action(probs)
        assert chosen_move in state.valid_moves(), (
            "Selected an invalid move, this should not happen."
        )
        return chosen_move


class Mlp(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = torch.nn.Linear(9, 128)
        self.fc1 = torch.nn.Linear(128, 128)
        self.fc2 = torch.nn.Linear(128, 128)
        self.unembed = torch.nn.Linear(128, 9)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.embed(x))
        x = torch.relu(self.fc1(x)) + x
        x = torch.relu(self.fc2(x)) + x
        x = self.unembed(x)
        return x


def reward_trajectory(trajectory: Trajectory, gamma: float) -> dict[Player, float]:
    total_rewards: dict[Player, float] = {"X": 0.0, "O": 0.0}
    discount = 1.0
    for _t, (_state, _action, reward, _prob) in enumerate(trajectory):
        total_rewards["X"] += reward["X"] * discount
        total_rewards["O"] += reward["O"] * discount
        discount *= gamma
    return total_rewards


def trajectory_surrogate(trajectory: Trajectory, gamma: float) -> torch.Tensor:
    reward = reward_trajectory(trajectory, gamma)
    # print(f"Trajectory reward: {reward}")
    x_probs = torch.stack(
        [
            step[3][to_flat(step[1])]
            for step in trajectory
            if step[0].current_player == "X"
        ]
    )
    o_probs = torch.stack(
        [
            step[3][to_flat(step[1])]
            for step in trajectory
            if step[0].current_player == "O"
        ]
    )
    log_x_probs = torch.log(x_probs + 1e-10)
    log_o_probs = torch.log(o_probs + 1e-10)
    x_surrogate = reward["X"] * log_x_probs.sum()
    o_surrogate = reward["O"] * log_o_probs.sum()
    # print(f"X surrogate: {x_surrogate.item()}, O surrogate: {o_surrogate.item()}")
    # surrogate = reward["X"] * log_x_probs.sum() + reward["O"] * log_o_probs.sum()
    return x_surrogate + o_surrogate


@dataclass
class H2HResult:
    x_wins: int
    o_wins: int
    x_draws: int
    o_draws: int
    x_losses: int
    o_losses: int


def h2h(
    policy1: Policy,
    policy2: Policy,
    num_games: int = 5000,
    policy1_name: str = "Policy 1",
    policy2_name: str = "Policy 2",
) -> tuple[H2HResult, H2HResult]:
    result1 = H2HResult(0, 0, 0, 0, 0, 0)
    result2 = H2HResult(0, 0, 0, 0, 0, 0)

    # policy 1 plays as X, policy 2 as O
    for _game_idx in range(num_games):
        state = State(current_player="X")
        while state.check_winner() == "NotFinished":
            if state.current_player == "X":
                action = policy1.select_action(state)
            else:
                action = policy2.select_action(state)
            state = state.transition(action)

        winner = state.check_winner()
        if winner == "X":
            result1.x_wins += 1
            result2.o_losses += 1
        elif winner == "O":
            result1.x_losses += 1
            result2.o_wins += 1
        else:
            result1.x_draws += 1
            result2.o_draws += 1

    # policy 2 plays as X, policy 1 as O
    for _game_idx in range(num_games):
        state = State(current_player="X")
        while state.check_winner() == "NotFinished":
            if state.current_player == "X":
                action = policy2.select_action(state)
            else:
                action = policy1.select_action(state)
            state = state.transition(action)

        winner = state.check_winner()
        if winner == "X":
            result2.x_wins += 1
            result1.o_losses += 1
        elif winner == "O":
            result2.x_losses += 1
            result1.o_wins += 1
        else:
            result2.x_draws += 1
            result1.o_draws += 1

    print("-" * 80)
    print(f"{policy1_name} winrate as X: {result1.x_wins / num_games:.2f}")
    print(f"{policy1_name} winrate as O: {result1.o_wins / num_games:.2f}")
    print(f"{policy1_name} drawrate as X: {result1.x_draws / num_games:.2f}")
    print(f"{policy1_name} drawrate as O: {result1.o_draws / num_games:.2f}")
    print("-" * 80)

    print(f"{policy2_name} winrate as X: {result2.x_wins / num_games:.2f}")
    print(f"{policy2_name} winrate as O: {result2.o_wins / num_games:.2f}")
    print(f"{policy2_name} drawrate as X: {result2.x_draws / num_games:.2f}")
    print(f"{policy2_name} drawrate as O: {result2.o_draws / num_games:.2f}")
    print("-" * 80)

    return result1, result2
