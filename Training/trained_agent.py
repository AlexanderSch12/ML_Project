#!/usr/bin/env python3
# encoding: utf-8
"""
dotsandboxes_agent.py
Extend this class to provide an agent that can participate in a tournament.
Created by Pieter Robberechts, Wannes Meert.
Copyright (c) 2022 KU Leuven. All rights reserved.
"""

import sys
import logging
from absl import app
from absl import flags

import numpy as np
import pyspiel
import training_dqn

from open_spiel.python import rl_environment
from open_spiel.python.algorithms import random_agent
from open_spiel.python.algorithms import evaluate_bots


FLAGS = flags.FLAGS

# Training parameters

flags.DEFINE_string("checkpoint_dir", "dqn_dnb_model_15x15.pt",
                    "Directory to save/load the agent models.")
flags.DEFINE_integer(
    "save_every", int(1e3),
    "Episode frequency at which the DQN agent models are saved.")
flags.DEFINE_integer("num_train_episodes", int(1e6),
                     "Number of training episodes.")
flags.DEFINE_integer(
    "eval_every", 100,
    "Episode frequency at which the DQN agents are evaluated.")

# DQN model hyper-parameters
flags.DEFINE_integer("hidden_layers_sizes", 64,
                  "Number of hidden units in the Q-Network MLP.")
flags.DEFINE_integer("replay_buffer_capacity", int(1e4),
                     "Size of the replay buffer.")
flags.DEFINE_integer("batch_size", 32,
                     "Number of transitions to sample at each learning step.")


def eval_against_random_bots(env, trained_agents, random_agents, num_episodes):
  """Evaluates `trained_agents` against `random_agents` for `num_episodes`."""
  num_players = len(trained_agents)
  sum_episode_rewards = np.zeros(num_players)
  for player_pos in range(num_players):
    cur_agents = random_agents[:]
    cur_agents[player_pos] = trained_agents[player_pos]
    for _ in range(num_episodes):
      time_step = env.reset()
      episode_rewards = 0
      while not time_step.last():
        player_id = time_step.observations["current_player"]
        if env.is_turn_based:
          agent_output = cur_agents[player_id].step(
              time_step, is_evaluation=True)
          action_list = [agent_output.action]
        else:
          agents_output = [
              agent.step(time_step, is_evaluation=True) for agent in cur_agents
          ]
          action_list = [agent_output.action for agent_output in agents_output]
        time_step = env.step(action_list)
        episode_rewards += time_step.rewards[player_pos]
      sum_episode_rewards[player_pos] += episode_rewards
  return sum_episode_rewards / num_episodes


def main(argv=None):
  game_string = "dots_and_boxes(num_rows=15,num_cols=15)"
  num_players = 2

  env_configs = {}
  env = rl_environment.Environment(game_string, **env_configs)
  info_state_size = env.observation_spec()["info_state"][0]
  num_actions = env.action_spec()["num_actions"]

  # random agents for evaluation
  random_agents = [
    random_agent.RandomAgent(player_id=idx, num_actions=num_actions)
    for idx in range(num_players)
  ]

  agents = [
    training_dqn.DQN(
      player_id=idx,
      state_representation_size=info_state_size,
      num_actions=num_actions,
      hidden_layers_sizes=FLAGS.hidden_layers_sizes,
      replay_buffer_capacity=FLAGS.replay_buffer_capacity,
      batch_size=FLAGS.batch_size) for idx in range(num_players)
  ]   

  for ep in range(FLAGS.num_train_episodes):
        if (ep + 1) % FLAGS.eval_every == 0:
          r_mean = eval_against_random_bots(env, agents, random_agents, 1000)
          logging.info("[%s] Mean episode rewards %s", ep + 1, r_mean)
        if (ep + 1) % FLAGS.save_every == 0:
          for agent in agents:
            agent.save(FLAGS.checkpoint_dir)

        time_step = env.reset()
        while not time_step.last():
          player_id = time_step.observations["current_player"]
          agent_output = agents[player_id].step(time_step)
          action_list = [agent_output.action]
          time_step = env.step(action_list)

        # Episode is over, step all agents with final info state.
        for agent in agents:
          agent.step(time_step)


if __name__ == "__main__":
  app.run(main)
