import math
import random
from utils.constraint import reflect

min_value = 0
max_value = 100

for value in [random.uniform(-250., 250.) for _ in range(10)]:
    print(value, "-->", reflect(value, min_value, max_value))

import matplotlib.pyplot as plt

# define constraint space
min_value = -50
max_value = 50
# generate points
n_points = 20
points = [(random.randint(-250, 250), random.randint(-250, 250)) for _ in range(n_points)]
# define figure
plt.figure(figsize=(8, 8))
plt.xlabel("x")
plt.ylabel("y")
plt.axvline(x=min_value, color='r', linestyle='-', linewidth=0.4)
plt.axvline(x=max_value, color='r', linestyle='-', linewidth=0.4)
plt.axhline(y=min_value, color='r', linestyle='-', linewidth=0.4)
plt.axhline(y=max_value, color='r', linestyle='-', linewidth=0.4)
for i, (x, y) in enumerate(points):
    new_x = reflect(x, min_value, max_value)
    new_y = reflect(y, min_value, max_value)
    plt.scatter(x, y, marker="x", color='g', label="original" if i == 0 else "")
    plt.scatter(new_x, new_y, marker="x", color='b', label="reflected" if i == 0 else "")
    plt.arrow(x=x, y=y, dx=new_x-x, dy=new_y-y)
plt.legend()
plt.savefig(fname="test/plot.png", format='png', transparent=False)
plt.show()