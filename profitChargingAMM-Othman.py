# https://www.cs.cmu.edu/~sandholm/profitChargingMarketMaker.ec12.pdf
import math
import matplotlib.pyplot as plt
import random
import numpy as np


###############################--Function Selection--###############################
print('Initial Set Up:')
print('Pick your barrier Utility Function -> u(x):')
print('\t 1) -1/x')
print('\t 2) log(x)')
choice_U = int(input('Pick a Number: '))
if choice_U == 1:
    def u(s):
        return -1/s
elif choice_U==2:
    def u(s):
        return math.log(s)

print('Pick your Liquidity Function -> f(s):')
print('\t 1) α*s**(1/b)')
print('\t 2) α*log(s+1)')
print('\t 3) Augur function: .924*((s + 132.3)**0.5 - 132.3**0.5)')
print('\t 4) Author experiment function: 100*(log(s + 1000) - log(1000))')
choice_F = int(input('Pick a Number: '))
if choice_F == 1:
    def f(s):
        return alpha*s**(1/beta)
elif choice_F==2:
    def f(s):
        return alpha*math.log(s+1)
elif choice_F==3:
    def f(s):
        return .924*((s + 132.3)**0.5 - 132.3**0.5)
elif choice_F==4:
    def f(s):
        return 100*(math.log(s + 1000) - 1000)


print('Pick your Profit Function -> g:')
print('\t 1) ε*x0*(1-1/(s+1))')
print('\t 2) Augur Profit Function: 0.01*s')
print('\t 3) Author experiment Function: 0,6*((s+100)**0.5-10)')
choice_G = int(input('Pick a Number: '))
if choice_G == 1:
    def g(x0, s):
        return epsilon*x0*(1-1/(s+1))
elif choice_G == 2:
    def g(s):
        return 0.001*s
elif choice_G == 3:
    def g(s):
        return 0,6*((s+100)**0.5-10)


###############################--Parameter choices--###############################
print('Parameter definition: ')

if choice_F == 1:
    print('Defining parameters for the Liquidity Function f:')
    alpha = float(input('\t Define a positive α (float): '))
    beta = float(input('\t Define a positive β (float): '))
elif choice_F == 2:
    print('Defining parameters for the Liquidity Function f:')
    alpha = float(input('\t Define a positive α (float): '))

if choice_G == 1:
    print('Defining parameters for the Profit Function g:')
    epsilon = float(input('\t Define a positive ε (float): '))

x0 = int(input('Define the initial wealth (author used 10) (int): '))

initial_price = float(input('Set an inicial price (float): '))

###############################--Solving Functions--###############################

def d(x, y):
    '''
    L1 norm. It sums the absolute distance between different x and y
    '''
    return sum(abs(x_i-y_i) for x_i, y_i in zip(x, y))

def inc_i(x, i):
    # returns a vector a single step in the +ith direction from x
    x_ = x[:]
    x_[i] += 1
    return x_

def C(x, ps, s, x0):
    '''
    This function solves implicitly the cost function. For this, the author (Othman)
    suggested using binary search.
    Equation: p_i*u(C(x) - x_i + f_s) = u(x0 + f_s)
    Left side of the equation: p_i*u(C(x) - x_i + f_s)
    Right side of the equation: u(x0 + f_s)
    Input:
        -x: cumulative amount the market maker must pay out to traders if future state of the world ω_i is realized.
        -ps: market maker’s (subjective) probability that ω_i will occur.
        -s: state. A measure of the cumulative volume transacted in the market
        -x0: initial wealth
    '''

    f_s = f(s)

    def left_side(C_x):
        '''
        This function resolves the left side of the cost equation
        '''
        result = 0
        for p_i, x_i in zip(ps, x):
            result += p_i*u(C_x - x_i + f_s)
        return result

    right_side = u(x0 + f_s)
    # lower bound for C(x) is max(x_i) - f_s
    lower_bound = max(x) - f_s
    upper_bound = max(x) + x0 + f_s
    approx = (lower_bound + upper_bound)/2
    left_side_approx = left_side(approx)
    eps = 1e-8
    steps = 1
    # binary search
    while abs(right_side - left_side_approx) > eps:
        if lower_bound == upper_bound:
            raise ValueError(
                "Failed to converge! stuck at %f, off by %f after %d steps" % (
                    approx, right_side - left_side_approx, steps))
        elif left_side_approx > right_side:
            upper_bound = approx
        else:
            lower_bound = approx
        approx = (lower_bound + upper_bound)/2
        left_side_approx = left_side(approx)
        steps += 1
    

    if choice_G == 1:
        return approx, g(x0,s), f_s, steps
    elif choice_G == 2:
        return approx, g(s), f_s, steps

def abe_msr(x, y, ps, x0, s):
    Cx, gs, fs, steps = C(x, ps, s, x0)
    Cy, gsd, fsd, steps = C(y, ps, s + d(x,y), x0)

    # try:
    #     print(f'total number of steps: {steps} : {d(point, point_i)}')
    # except:
    #     print(f'total number of steps: {steps}')

    return Cy - Cx, gsd - gs, fsd - fs, steps


###############################--Result plot--###############################

def make_plot(odds, point, s, x0):
    # Plots the change in the price of a fixed size bet
    # while varying s (market volume/ distance the payout vector
    # has been moved by trades)

    price_sum = [0] * len(s) # an array to keep track of the sum of the prices

    # set the axis labels on the plot
    plt.xlabel('$s$ ~ Volume') 
    plt.ylabel('$M(\mathbf{x}, \mathbf{y})$ ~ Price Per Share')
    # we vary each dimension of the point...
    listofpoints=[]
    for i in range(len(point)):
        point_i = inc_i(point, i) # ... by incrementing the ith value.
        label = '%s -> %s' % (tuple(point), tuple(point_i)) # label for the ith curve
        prices = [sum(abe_msr(point, point_i, odds, x0, si)) for si in s] # calculate price for varying s
        for j, pi in enumerate(prices):
            price_sum[j] += pi # add the price with each s to the price sum
        plt.semilogx(s, prices, label=label) # plot the curve
        print(point)
    plt.semilogx(s, price_sum, label='sum') # plot the sum
    # plt.semilogx(s, listofpoints, labes='price')
    plt.legend(loc=5) # create and set the legend location
    fname = './PCAMM-plots/odds - %s; x = %s, x0 = %d.png' % (tuple(odds), tuple(point), x0) # give a descriptive filename
    plt.title(fname)
    plt.savefig(fname)
    plt.close()


def variationPrice(initialPrice, length, maxPercVariation):
    price = initialPrice
    finalPrices = []
    for variation in range(0,length):
        variationList = list(np.random.normal(0, maxPercVariation, length))
        percVariation = 1+(maxPercVariation/100)
        newprice = round(price * percVariation,2)
        
        finalPrices += [newprice]
        price = newprice
    
    return finalPrices

if __name__ == '__main__':
    oddses = [[.5,.5], [.85,.15]]
    point = [0, 0]
    s = [10**(3.*i/50) for i in range(101)]
    for i, odds in enumerate(oddses):
        make_plot(odds, point, s, x0, )
        make_plot(odds, point, s, 100)
        make_plot(odds, point, s, 1000)

    oddses = [[.5, .3, .2], [.3, .25, .2, .15, .1]]
    #Probar qué pasaría si acá genero una serie de posibilidades armadas en forma aleatoria
    #en forma de que la suma de los valores de 1

    points = variationPrice(initial_price, len(oddses), 10)

    for odds in oddses:
        point = variationPrice(initial_price, len(odds), 10)
        # point = random_point(len(odds), 100)
        make_plot(odds, point, s, x0)

    # for i in range(0, len(oddses)):
    #     make_plot(oddses[i], points, s, x0)
