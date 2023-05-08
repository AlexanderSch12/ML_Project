#!/usr/bin/env python3
# encoding: utf-8
"""
dotsandboxes_agent.py
Extend this class to provide an agent that can participate in a tournament.
Created by Pieter Robberechts, Wannes Meert.
Copyright (c) 2022 KU Leuven. All rights reserved.
"""

import os
import random
import sys
import importlib.util
import logging
from absl import app
from absl import flags
import numpy as np
from pathlib import Path
import pyspiel
import tensorflow as tf

from open_spiel.python import rl_environment
from open_spiel.python.pytorch import dqn
from open_spiel.python.algorithms import random_agent

from dqn import DQN

logger = logging.getLogger('be.kuleuven.cs.dtai.dotsandboxes')


FLAGS = flags.FLAGS

# Training parameters
flags.DEFINE_string("checkpoint_dir", "/tmp/dqn_dnb_model",
                    "Directory to save/load the agent models.")
flags.DEFINE_integer(
    "save_every", int(1e4),
    "Episode frequency at which the DQN agent models are saved.")
flags.DEFINE_integer("num_train_episodes", int(1e6),
                     "Number of training episodes.")
flags.DEFINE_integer(
    "eval_every", 1000,
    "Episode frequency at which the DQN agents are evaluated.")

# DQN model hyper-parameters
flags.DEFINE_integer("hidden_layers_sizes", 64,
                  "Number of hidden units in the Q-Network MLP.")
flags.DEFINE_integer("replay_buffer_capacity", int(1e5),
                     "Size of the replay buffer.")
flags.DEFINE_integer("batch_size", 32,
                     "Number of transitions to sample at each learning step.")


def get_agent_for_tournament(player_id):
    """Change this function to initialize your agent.
    This function is called by the tournament code at the beginning of the
    tournament.
    :param player_id: The integer id of the player for this bot, e.g. `0` if
        acting as the first player.
    """
    agent = Agent(player_id)
    return agent


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

        # create env for real game size
        self.game_string = "dots_and_boxes(num_rows=4,num_cols=4)"
        env_configs = {}
        self.env = rl_environment.Environment(self.game_string, **env_configs)
        info_state_size = self.env.observation_spec()["info_state"][0]
        num_actions = self.env.action_spec()["num_actions"]

        # create env for trained 15x15 game size
        self.game_string_trained = "dots_and_boxes(num_rows=15,num_cols=15)"
        env_configs_trained = {}
        self.env_trained = rl_environment.Environment(self.game_string_trained, **env_configs_trained)
        info_state_size_trained = self.env_trained.observation_spec()["info_state"][0]
        num_actions_trained = self.env_trained.action_spec()["num_actions"]

        # TODO create illegal action representation based on size of real board an give it to the DQN agent
        illegal_actions = []

        # create trained 15x15 agent
        self.agent = DQN(
            player_id=player_id,
            state_representation_size=info_state_size_trained,
            num_actions=num_actions_trained,
            illegal_actions=illegal_actions,
            hidden_layers_sizes=FLAGS.hidden_layers_sizes,
            replay_buffer_capacity=FLAGS.replay_buffer_capacity,
            batch_size=FLAGS.batch_size)

        if self.agent.has_checkpoint("./dqn_dnb_model"):
            print("checkpoint found!")
        self.agent.load("./dqn_dnb_model")


    def restart_at(self, state):
        """Starting a new game in the given state.
        :param state: The initial state of the game.
        """
        # self.env.set_state(state)
        self.game = state.get_game()
        self.env.reset()
        self.env_trained.reset()


    def inform_action(self, state, player_id, action):
        """Let the bot know of the other agent's actions.
        :param state: The current state of the game.
        :param player_id: The ID of the player that executed an action.
        :param action: The action which the player executed.
        """
        # inform real size environment of action
        self.env.step([action])

        # inform 15x15 size environmnet of action
        # TODO convert action and infrom env_trained

        
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
            # TODO convert agent_output back to real size board action and apply to env

            # apply action to env_trained
            self.env_trained.step([agent_output.action])
        else:
            agent_output = self.agent.step(time_step, is_evaluation=True)

        return agent_output.action


def evaluate_bots(state, bots, rng):
  """Plays bots against each other, returns terminal utility for each bot."""
  for bot in bots:
    bot.restart_at(state)
  print(state)
  while not state.is_terminal():
    current_player = state.current_player()
    print("Move by player: " + str(current_player))
    action = bots[current_player].step(state)
    for i, bot in enumerate(bots):
        if i != current_player:
            bot.inform_action(state, current_player, action)
    state.apply_action(action)
    print(state)
  return state.returns()


class UniformRandomBot(pyspiel.Bot):

    def __init__(self, player_id, rng):
        pyspiel.Bot.__init__(self)
        self._player_id = player_id
        self._rng = rng

    def inform_action(self, state, player_id, action):
        pass

    def restart_at(self, state):
        pass

    def player_id(self):
        return self._player_id

    def provides_policy(self):
        return True

    def step_with_policy(self, state):
        legal_actions = state.legal_actions(self._player_id)
        if not legal_actions:
            return [], pyspiel.INVALID_ACTION
        p = 1 / len(legal_actions)
        policy = [(action, p) for action in legal_actions]
        action = self._rng.choice(legal_actions)
        return policy, action

    def step(self, state):
        return self.step_with_policy(state)[1]
  

def test_api_calls():
    """This method calls a number of API calls that are required for the
    tournament. It should not trigger any Exceptions.
    """
    dotsandboxes_game_string = (
        "dots_and_boxes(num_rows=4,num_cols=4)")
    game = pyspiel.load_game(dotsandboxes_game_string)
    logger.info("Loading the agents")
    bots = [get_agent_for_tournament(0), UniformRandomBot(player_id=1, rng=np.random)]
    returns = evaluate_bots(game.new_initial_state(), bots, np.random)
    print("-----------------------------------------------")
    print(returns)
    assert len(returns) == 2
    assert isinstance(returns[0], float)
    assert isinstance(returns[1], float)
    print("SUCCESS!")


def main(argv=None):
    test_api_calls()


if __name__ == "__main__":
    app.run(main)