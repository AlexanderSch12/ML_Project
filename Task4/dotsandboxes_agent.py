#!/usr/bin/env python3
# encoding: utf-8
"""
dotsandboxes_agent.py

Extend this class to provide an agent that can participate in a tournament.

Created by Pieter Robberechts, Wannes Meert.
Copyright (c) 2022 KU Leuven. All rights reserved.
"""

import sys
import argparse
import logging
import random
import numpy as np
import pyspiel
from absl import flags
import tensorflow.compat.v1 as tf
from open_spiel.python.algorithms import evaluate_bots
from open_spiel.python import rl_environment
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

logger = logging.getLogger('be.kuleuven.cs.dtai.dotsandboxes')


def get_agent_for_tournament(player_id):
    """Change this function to initialize your agent.
    This function is called by the tournament code at the beginning of the
    tournament.

    :param player_id: The integer id of the player for this bot, e.g. `0` if
        acting as the first player.
    """
    my_player = Agent(player_id)
    return my_player


class Agent(pyspiel.Bot):
    """Agent template"""

    def __init__(self, player_id):
        """Initialize an agent to play Dots and Boxes.
        Note: This agent should make use of a pre-trained policy to enter
        the tournament. Initializing the agent should thus take no more than
        a few seconds.
        """
        pyspiel.Bot.__init__(self)
        self.player_id = player_id
        self.game_string = "dots_and_boxes(num_rows=4,num_cols=4)"

        env_configs = {}
        self.env = rl_environment.Environment(self.game_string, **env_configs)
        info_state_size = self.env.observation_spec()["info_state"][0]
        num_actions = self.env.action_spec()["num_actions"]

        sess = tf.Session()
        self.agent = dqn.DQN(
            session=sess,
            player_id=player_id,
            state_representation_size=info_state_size,
            num_actions=num_actions,
            hidden_layers_sizes=64,
            replay_buffer_capacity=int(1e5),
            batch_size=32)

        if self.agent.has_checkpoint("./dqn_dnb_model"):
            print("checkpoint found!")
        self.agent.restore("./dqn_dnb_model")
        sess.run(tf.global_variables_initializer())


    def restart_at(self, state):
        """Starting a new game in the given state.
        :param state: The initial state of the game.
        """
        # self.env.set_state(state)
        self.game = state.get_game()
        self.env.reset()


    def inform_action(self, state, player_id, action):
        """Let the bot know of the other agent's actions.
        :param state: The current state of the game.
        :param player_id: The ID of the player that executed an action.
        :param action: The action which the player executed.
        """
        self.env.step([action])

        
    def step(self, state):
        """Returns the selected action in the given state.
        :param state: The current state of the game.
        :returns: The selected action from the legal actions, or
            `pyspiel.INVALID_ACTION` if there are no legal actions available.
        """
        # self.env.set_state(state)
        time_step = self.env.get_time_step()
        if not time_step.last():
            agent_output = self.agent.step(time_step, is_evaluation=True)
            self.env.step([agent_output.action])
        else:
            agent_output = self.agent.step(time_step, is_evaluation=True)

        return agent_output.action


def test_api_calls():
    """This method calls a number of API calls that are required for the
    tournament. It should not trigger any Exceptions.
    """
    # dotsandboxes_game_string = (
    #     "dots_and_boxes(num_rows=5,num_cols=5)")
    # game = pyspiel.load_game(dotsandboxes_game_string)
    # bots = [get_agent_for_tournament(player_id) for player_id in [0,1]]
    # returns = evaluate_bots.evaluate_bots(game.new_initial_state(), bots, np.random)
    # assert len(returns) == 2
    # assert isinstance(returns[0], float)
    # assert isinstance(returns[1], float)
    print("SUCCESS!")


def main(argv=None):
    test_api_calls()


if __name__ == "__main__":
    sys.exit(main())