from absl import app
import pyspiel

from open_spiel.python import rl_environment
from open_spiel.python.algorithms import random_agent


def test():
    game_string = "dots_and_boxes(num_rows=4,num_cols=4)"
    env_configs = {}
    env = rl_environment.Environment(game_string, **env_configs)
    env.reset()
    time_step = env.get_time_step()
    legal_actions = time_step.observations["legal_actions"][0]
    print(legal_actions)
    env.step([1])
    legal_actions = time_step.observations["legal_actions"][0]
    print(legal_actions)

    return


def main(argv=None):
    test()


if __name__ == "__main__":
    app.run(main)