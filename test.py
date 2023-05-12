from absl import app
import pyspiel

from open_spiel.python import rl_environment
from open_spiel.python.algorithms import random_agent

def transform_edge_number( rows, cols):
    result = []
    
    # Vertical
    for r in range(rows):
        for c in range(cols+1):
            result.append((r*6)+30 + c)

    # Horizontal
    for r in range(rows+1):
        for c in range(cols):
            result.append((r*5) + c)
        
    result.sort()
    return result


def test():
    edges_3x2 = transform_edge_number(3, 2)
    print(edges_3x2)

    return


def main(argv=None):
    test()


if __name__ == "__main__":
    app.run(main)