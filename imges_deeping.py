
import numpy as np

b = [37.8,39.3,45.9,41.3]
a = [22.1,10.4,18.3,18.5]

# Y = Weight * x + Bias
# Y = MX +  B

def pricdict(a,weight,bias):
    return weight * a + bias

def cost_function(a,b,weight,bias):
    companies = len(a)
    total = 0.0
    for i in range(companies):
        total +=(b[i] - (weight * a[i] + bias)) ** 2
    return total / companies

def update_weight(a,b, weight, bias, learning_rate):
    weight_deriv = 0
    bias_deriv = 0
    companies = len(a)
    for i in range(companies):
        weight_deriv += -2 * (a[i] * (b[i]) - (weight * a[i] + bias))
        bias_deriv += -2 * (b[i]  - (weight * b[i] + bias))
    weight -= (weight_deriv / companies) * learning_rate
    bias -= (bias_deriv / companies) * learning_rate

    return weight, bias


def train(a, b, weight, bias, learning_rate, iters):
    cost_history = []

    for i in range(iters):
        weight,bias = update_weights(a, b, weight, bias, learning_rate)

        #Calculate cost for auditing purposes
        cost = cost_function(a, b, weight, bias)
        cost_history.append(cost)

        # Log Progress
        if i % 10 == 0:
            print("iter={:d}    weight={:.2f}    bias={:.4f}    cost={:.2}".format(i, weight, bias, cost))

    return weight, bias, cost_history        
        


