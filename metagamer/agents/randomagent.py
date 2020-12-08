import numpy as np
import random
from metagamer.environments import tictactoe


class BetterToe(tictactoe.TicTacToeEnv):
    def _get_obs(self) -> np.array:
        return self.board.flatten()


def main():
    game = BetterToe()

    all_games = []
    all_choices = []
    game_rewards = []

    for _ in range(10000):
        # whole new game
        game.reset()
        game_states = []
        choices = []
        while True:
            # player 1
            choice_1 = random.choice(game.valid_actions)
            obs, reward, done, _ = game.step(choice_1, 1)
            # game.render()

            game_states.append(obs)
            choices.extend(choice_1)

            if done:
                break

            # player 2
            choice_2 = random.choice(game.valid_actions)
            obs, reward, done, _ = game.step(choice_2, -1)
            # game.render()
            game_states.append(obs)
            choices.extend(choice_2)

            if done:
                break

        all_choices.append(tuple(choices))
        # curr_game = np.stack(game_states)

        # all_games.append()
        game_rewards.append(reward)

    print(set(all_choices))
    print(len(all_choices))

    print(all_games)
    print(game_rewards)


if __name__ == "__main__":
    main()
