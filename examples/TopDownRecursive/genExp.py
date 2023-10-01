import random as rand

# recursive production function
# used to expand strings generated by the grammar

# E -> TE1
def E():
    T()
    E1()

# E1 -> +TE1 | e
def E1():
    r = rand.random()
    if r<0.2:
        # select first production with p=0.2
        print('+',end='')
        T()
        E1()

# T -> FT1
def T():
    F()
    T1()

# T1 -> *FT1 | e
def T1():
    r = rand.random()
    if r<0.2:
        # select first production with p=0.2
        print('*',end='')
        F()
        T1()


# F -> (E) | n
def F():
    r = rand.random()
    if r<0.3:
        # select first production with p=0.2
        print('(',end='')
        E()
        print(')', end='')
    else:
        print('n', end='')

for i in range(10):
    E()
    print()