# Binary Genetic algorithm in Python

*Status* : under development

## What's New
version 0.0.1 : intial version.

## Presentation
In computer science and operations research, a **genetic algorithm** (GA) is a metaheuristic inspired by the process of natural selection that belongs to the larger class of evolutionary algorithms (EA). Genetic algorithms are commonly used to generate high-quality solutions to optimization and search problems by relying on bio-inspired operators such as mutation, crossover and selection.

## Prerequisites

The present code has been developed under python3.x. You will need to have the following installed on your computer to make it work :

* Python 3.x
* Numpy

## Using Binary Genetic algorithm (BGA)

The first thing you should do is to define your own evaluation function for your own optimization problem. For example, for 0-1 pakages problems i will define the evaluation function as follows:

```python
# arr is the individual vector
# w for weights
# v for values
# 1000 is the maximum weight.
def values(arr):
    w = np.array([71, 34, 82, 23, 1, 88, 12, 57, 10, 68, 5, 33, 37, 69, 98, 24, 26, 83, 16, 26, 18, 43, 52, 71, 22, 65, 68, 8, 40, 40, 24, 72, 16, 34, 10, 19, 28, 13, 34, 98, 29, 31, 79, 33, 60, 74, 44, 56, 54, 17, 63, 83, 100, 54, 10, 5, 79, 42, 65, 93, 52, 64, 85, 68, 54, 62, 29, 40, 35, 90, 47, 77, 87, 75, 39, 18, 38, 25, 61, 13, 36, 53, 46, 28, 44, 34, 39, 69, 42, 97, 34, 83, 8, 74, 38, 74, 22, 40, 7, 94])
    v = np.array([26, 59, 30, 19, 66, 85, 94, 8, 3, 44, 5, 1, 41, 82, 76, 1, 12, 81, 73, 32, 74, 54, 62, 41, 19, 10, 65, 53, 56, 53, 70, 66, 58, 22, 72, 33, 96, 88, 68, 45, 44, 61, 78, 78, 6, 66, 11, 59, 83, 48, 52, 7, 51, 37, 89, 72, 23, 52, 55, 44, 57, 45, 11, 90, 31, 38, 48, 75, 56, 64, 73, 66, 35, 50, 16, 51, 33, 58, 85, 77, 71, 87, 69, 52, 10, 13, 39, 75, 38, 13, 90, 35, 83, 93, 61, 62, 95, 73, 26, 85])
    w_ = np.sum(w * arr)
    if w_ > 1000:
#         print(np.sum(w * arr))
        return 1.0 / (w_ - 1000)
    else:
        return np.sum(v * arr)
```

After defining the evalution function, you should initialize the BGA class and use .run to find the opetimal solution.
```python
from bga import BGA
num_pop = 30
problem_dimentions = 10

test = BGA(pop_shape=(num_pop, problem_dimentions), method=values, p_c=0.8, p_m=0.2, max_round = 1000, early_stop_rounds=None, verbose = None, maximum=True)
best_solution, best_fitness = test.run()
```

The whole test code is shown as follows:
```python
import numpy as np
from bga import BGA

def values(arr):
    w = np.array([71, 34, 82, 23, 1, 88, 12, 57, 10, 68, 5, 33, 37, 69, 98, 24, 26, 83, 16, 26, 18, 43, 52, 71, 22, 65, 68, 8, 40, 40, 24, 72, 16, 34, 10, 19, 28, 13, 34, 98, 29, 31, 79, 33, 60, 74, 44, 56, 54, 17, 63, 83, 100, 54, 10, 5, 79, 42, 65, 93, 52, 64, 85, 68, 54, 62, 29, 40, 35, 90, 47, 77, 87, 75, 39, 18, 38, 25, 61, 13, 36, 53, 46, 28, 44, 34, 39, 69, 42, 97, 34, 83, 8, 74, 38, 74, 22, 40, 7, 94])
    v = np.array([26, 59, 30, 19, 66, 85, 94, 8, 3, 44, 5, 1, 41, 82, 76, 1, 12, 81, 73, 32, 74, 54, 62, 41, 19, 10, 65, 53, 56, 53, 70, 66, 58, 22, 72, 33, 96, 88, 68, 45, 44, 61, 78, 78, 6, 66, 11, 59, 83, 48, 52, 7, 51, 37, 89, 72, 23, 52, 55, 44, 57, 45, 11, 90, 31, 38, 48, 75, 56, 64, 73, 66, 35, 50, 16, 51, 33, 58, 85, 77, 71, 87, 69, 52, 10, 13, 39, 75, 38, 13, 90, 35, 83, 93, 61, 62, 95, 73, 26, 85])
    w_ = np.sum(w * arr)
    if w_ > 1000:
#         print(np.sum(w * arr))
        return 1.0 / (w_ - 1000)
    else:
        return np.sum(v * arr)

num_pop = 30
problem_dimentions = 10

test = BGA(pop_shape=(num_pop, problem_dimentions), method=values, p_c=0.8, p_m=0.2, max_round = 1000, early_stop_rounds=None, verbose = None, maximum=True)
best_solution, best_fitness = test.run()
```

The output for test case is shown as follows:
```
Did not improved within 100 rounds. Break.

 Solution: [0 0 0 1 1 0 1 0 0 0 1 0 0 1 0 0 0 0 1 0 1 0 0 0 1 0 0 1 0 0 1 1 1 0 1 1 1
 1 0 0 1 1 0 1 0 0 0 0 1 1 0 0 0 0 1 1 0 0 0 0 0 0 0 1 0 0 0 1 0 0 0 0 0 0
 1 1 1 0 0 1 1 0 0 1 0 0 0 0 0 0 1 0 1 0 1 1 1 0 1 0]
 Fitness: 2325
 Evaluation times: 20730
```
