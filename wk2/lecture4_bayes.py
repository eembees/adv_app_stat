import numpy as np

# params

p_true_1 = 0.8
p_true_2 = 0.3
p1 = 0.1
p2 = 0.9


p_true = p1 * p_true_1 + p2*p_true_2

p_1_true = p_true_1 * p1 / p_true

print("Prob true given positive")
print(p_1_true)

# Decrease FP by factor 2

p_true_2 = 0.15

p_true = p1 * p_true_1 + p2*p_true_2

p_1_true = p_true_1 * p1 / p_true

print("Decrease FP")
print(p_1_true)


p_true_2 = 0.3
p1 = 0.16
p2 = 1 - p1

p_true = p1 * p_true_1 + p2*p_true_2

p_1_true = p_true_1 * p1 / p_true

print("Increase selection")
print(p_1_true)

