# ML Project February 2023: Alexander Schoeters & Maarten Kesters

This directory contains the files that we created for the ML Project: Multi-Agent Learning in Canonical Games and Dots-and-Boxes (OpenSpiel)

## Available files

- `dotsandboxes_agent.py`: Code used to take part in the tournament
- `dqn_2.py`: Adapted dqn agent code from Openspiel used by our agent in the tournament
- `dqn_dnb_model_0_15x15.pt`: Trained model for the agent that plays the first move
- `dqn_dnb_model_1_15x15.pt`: Trained model for the agent that plays the second move
- `trained_agent.py`: Code used to train our model
- `training_dqn.py`: Adapted dqn agent code from Openspiel used to train our model
- `ML_Project_Task2.ipynb`: Code for part 2 of the project about learning and dynamics
- `part3.py`: Code for part 3 of the project about Minimax for small Dots-and-Boxes

## Part 2

Notebook `ML_Project_Task2.ipynb` can be used to create the plots for the learning trajectories and the dynamics of the 4 benchmark matrix games.

## Part 3

File `part3.py` Can be used to run the Minimax algorithm for small Dots-and-Boxes games. At the top of the file, the number of rows and columns of the board can set. The code uses both a transposition table and symmetries.

## Part 4

`dotsandboxes_agent.py` contains the Agent class used for the tournament and can be called from get_agent_for_tournament(player_id). Running `dotsandboxes_agent.py`, plays 20 games of dots and boxes on a 7x7 board against a random bot. It will use the models `dqn_dnb_model_15x15_0.pt` and `dqn_dnb_model_15x15_1.pt`.

### Training

Training the model can be done by running `trained_agent.py`. This will train and evaluate a model for dots and boxes on a 15x15-board. Two models will be saved: `dqn_dnb_model_15x15_0.pt` and `dqn_dnb_model_15x15_1.pt`. One for the agent who has the first move and one for the agent who has the second move. They will be saved for a first time after 1000 episodes (overwriting the previous models).

### Tournament

`dotsandboxes_agent.py` contains the Agent class used for the tournament. It can be called from get_agent_for_tournament(player_id).
