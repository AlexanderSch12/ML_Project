import numpy as np
import matplotlib.pyplot as plt

# Define the parameters for ε-greedy and Lenient Boltzmann Q-learning
epsilon = 0.1
temperature = 0.1
temp_decay = 0.99
min_temp = 0.01
leniency = 0.1


def epsilon_greedy(Q, state, num_actions):
    if np.random.uniform(0, 1) < epsilon:
        # Choose a random action
        a = np.random.randint(0, num_actions)
    else:
        # Choose the action with the highest Q-value
        # a = max(range(num_actions), key=lambda x: Q[x])
        a = np.argmax(Q[state])
    return a


def lenient_boltzmann(Q, state, temp, num_actions):
    # Compute the lenient probabilities for each action
    lenient_probs = np.exp(Q[state] / temperature)
    lenient_probs = lenient_probs / np.sum(lenient_probs)
    max_q = np.max(Q[state])

    # Compute the strict probabilities for each action
    strict_probs = np.zeros(num_actions)
    strict_probs[np.where(Q[state] == max_q)] = 1 / np.sum(Q[state] == max_q)

    # Combine the lenient and strict probabilities using the leniency factor
    probs = leniency * lenient_probs + (1 - leniency) * strict_probs

    # Choose an action according to the probabilities
    a = np.random.choice(num_actions, p=probs)

    return a


def update_Q(Q, s, a, r, s_next, alpha, gamma):
    Q[s][a] += alpha * (r + gamma * np.max(Q[s_next]) - Q[s][a])
    return Q


# ----------------------------- Nash Equilibrium ----------------------------- #
def nash_equilibrium(game, Q_table):
    # Check for convergence to Nash equilibrium
    nash_eq = np.array([np.argmax(Q_table[0]), np.argmax(Q_table[1])])
    print("Nash equilibrium:", nash_eq)
    if np.allclose(Q_table[nash_eq[0]][nash_eq[1]], game[nash_eq[0]][nash_eq[1]], rtol=1e-3):
        print("Converged to Nash equilibrium!")
    else:
        print("Did not converge to Nash equilibrium.")


# ----------------------------- Mixed Equilibrium ----------------------------- #
def mixes_equilibrium(Q_table, num_players, num_actions):
    probs = []
    for i in range(num_players):
        if np.random.uniform(0, 1) < epsilon:
            # Choose a random action with probability epsilon
            action_probs = np.ones(num_actions) / num_actions
        else:
            # Choose the action with the highest Q-value with probability 1-epsilon
            best_actions = np.argwhere(Q_table[i] == np.max(Q_table[i])).flatten()
            action_probs = np.zeros(num_actions)
            action_probs[best_actions] = 1.0 / len(best_actions)
        probs.append(action_probs)
    print(probs)


def trajectory_plot(num_episodes, player1_cumulative_rewards, player2_cumulative_rewards):
    # Plot the learning trajectories
    plt.plot(range(num_episodes), player1_cumulative_rewards, label='Player 1')
    plt.plot(range(num_episodes), player2_cumulative_rewards, label='Player 2')
    plt.xlabel('Episodes')
    plt.ylabel('Cumulative Rewards')
    plt.title('Empirical Learning Trajectories in Prisoner\'s Dilemma')
    plt.legend()
    plt.show()


# ----------------------------- E-Greedy / Lenient-Boltzmann ----------------------------- #
def play_game(game, num_actions, num_players, num_episodes, alpha, greedy):
    # Initialize Q-values for each action
    Q_table = np.zeros((num_actions, num_actions))

    # Initialize empty lists to store rewards obtained by each player
    player1_rewards = []
    player2_rewards = []

    # play multiple episodes of the game

    for i in range(num_episodes):
        # Choose actions for both players using epsilon-greedy algorithm
        if greedy:
            p1_action = epsilon_greedy(Q_table, 0, num_actions)
            p2_action = epsilon_greedy(Q_table, 1, num_actions)
        else:
            p1_action = lenient_boltzmann(Q_table, 0, temperature, num_actions)
            p2_action = lenient_boltzmann(Q_table, 1, temperature, num_actions)

        # Determine rewards obtained by both players in the current episode
        p1_reward = game[0][p1_action][p2_action]
        p2_reward = game[1][p1_action][p2_action]

        # Append rewards obtained by each player to their respective lists
        player1_rewards.append(p1_reward)
        player2_rewards.append(p2_reward)

        # Update Q-values using the reinforcement learning rule
        Q_table[0][p1_action] += alpha * (game[0][p1_action][p2_action] - Q_table[0][p1_action])
        Q_table[1][p2_action] += alpha * (game[1][p1_action][p2_action] - Q_table[1][p2_action])

    # Print the learned Q-values for each action
    print("Learned Q-values:")
    print(Q_table)

    # Compute cumulative rewards obtained by each player over the course of the episodes
    player1_cumulative_rewards = np.cumsum(player1_rewards)
    player2_cumulative_rewards = np.cumsum(player2_rewards)

    # print(player1_cumulative_rewards)
    trajectory_plot(num_episodes, player1_cumulative_rewards, player2_cumulative_rewards)
    return Q_table


def main():
    print("######## Prisoners Dilemma 1 ########")
    PD_game1 = np.array([[[-1, -1], [1, 1]],
                         [[1, 1], [-1, -1]]])
    print("E-Greedy:")
    q_table = play_game(PD_game1, 2, 2, 1000, 0.1, True)
    nash_equilibrium(PD_game1, q_table)
    print()

    print("Lenient-Boltzmann:")
    q_table = play_game(PD_game1, 2, 2, 1000, 0.1, False)
    nash_equilibrium(PD_game1, q_table)
    print()

    print("######## Battle of Sexes ########")
    BOS_game = np.array([[[3, 2], [0, 0]],
                         [[0, 0], [2, 3]]])
    print("E-Greedy:")
    q_table = play_game(BOS_game, 2, 2, 1000, 0.1, True)
    nash_equilibrium(BOS_game, q_table)
    print()

    print("Lenient-Boltzmann:")
    q_table = play_game(BOS_game, 2, 2, 1000, 0.1, False)
    nash_equilibrium(BOS_game, q_table)
    print()

    print("######## Prisoners Dilemma 2 ########")
    PD_game2 = np.array([[[-1, -1], [-4, 0]],
                         [[0, -4], [-3, -3]]])
    print("E-Greedy:")
    q_table = play_game(PD_game2, 2, 2, 1000, 0.1, True)
    nash_equilibrium(PD_game2, q_table)
    print()

    print("Lenient-Boltzmann:")
    q_table = play_game(PD_game2, 2, 2, 1000, 0.1, False)
    nash_equilibrium(PD_game2, q_table)
    print()

    print("######## Rock-Paper-Scissors ########")
    RPS_game = np.array(
        [[[0, -0.25, 0.5],
          [0.25, 0, -0.05],
          [-0.5, 0.05, 0]],
         [[0, 0.25, -0.5],
          [-0.25, 0, 0.05],
          [0.5, -0.05, 0]]])
    print("E-Greedy:")
    q_table = play_game(RPS_game, 3, 2, 1000, 0.1, True)
    nash_equilibrium(RPS_game, q_table)
    print()

    print("Lenient-Boltzmann:")
    q_table = play_game(RPS_game, 3, 2, 1000, 0.1, False)
    nash_equilibrium(RPS_game, q_table)


if __name__ == "__main__":
    main()
