import numpy as np

def step(x):
  if x>=0:
    return 1
  else:
    return 0
def percep(x,w,b):
  v = np.dot(x,w)+b
  y = step(v)
  return y

def and_gate(x):
  w = np.array([1,1])
  b = -2
  return percep(x,w,b)

w = np.array([1,1])
b = -2

test1 = np.array([0,0])
test2 = np.array([1,0])
test3 = np.array([0,1])
test4 = np.array([1,1])

print(f"AND({0},{0})={and_gate(test1)}")
print(f"AND({0},{1})={and_gate(test2)}")
print(f"AND({1},{0})={and_gate(test3)}")
print(f"AND({1},{1})={and_gate(test4)}")
print(f"Final weight:{w}")
print(f"Final bias:{b}")

def activation_func(v):
  if v >=0:
    return 1
  else:
    return 0


def perceptron(w,x,b):
  v = np.dot(w,x)+b
  y = activation_func(v)
  return y


def or_gate(x):
  w = np.array([1,1])
  b = -1
  return perceptron(w,x,b)


test1 = np.array([0, 0])
test2 = np.array([0, 1])
test3 = np.array([1, 1])
test4 = np.array([1, 0])



print(f"OR({0},{0})={or_gate(test1)}")
print(f"OR({0},{1})={or_gate(test2)}")
print(f"OR({1},{0})={or_gate(test3)}")
print(f"OR({1},{1})={or_gate(test4)}")

def activation_func(v):
  if v >=0:
    return 1
  else:
    return 0

def perceptron(w,x,b):
  v = np.dot(w,x)+b
  y = activation_func(v)
  return y

def NOT_GATE(x):
  w = np.array([-1])
  b = 0.5
  return perceptron(w,x,b)

test1 = np.array([0])
test2 = np.array([1])

print(f"Not gate of ({test1}) = ",NOT_GATE(test1))
print(f"Not gate of ({test2}) = ",NOT_GATE(test2))

def NOR_GATE(x):
  y = or_gate(x)
  v = NOT_GATE(y)
  return v

test1 = np.array([0, 0])
test2 = np.array([0, 1])
test3 = np.array([1, 0])
test4 = np.array([1, 1])

print(f"NOR gate of ({test1}) = ",NOR_GATE(test1))
print(f"NOR gate of ({test2}) = ",NOR_GATE(test2))
print(f"NOR gate of ({test3}) = ",NOR_GATE(test3))
print(f"NOR gate of ({test4}) = ",NOR_GATE(test4))

def NAND_GATE(x):
  y = and_gate(x)
  v = NOT_GATE(y)
  return v


test1 = np.array([0, 0])
test2 = np.array([0, 1])
test3 = np.array([1, 0])
test4 = np.array([1, 1])



print(f"NAND gate of ({test1}) = ",NAND_GATE(test1))
print(f"NAND gate of ({test2}) = ",NAND_GATE(test2))
print(f"NAND gate of ({test3}) = ",NAND_GATE(test3))
print(f"NAND gate of ({test4}) = ",NAND_GATE(test4))

def XOR_GATE(x):
  y1=and_gate(x)
  y2=or_gate(x)
  y3=NOT_GATE(y1)
  final_x=np.array([y2,y3])
  y=and_gate(final_x)
  return y

test1 = np.array([0, 0])
test2 = np.array([0, 1])
test3 = np.array([1, 0])
test4 = np.array([1, 1])

print(f"XOR gate of ({test1}) = ",XOR_GATE(test1))
print(f"XOR gate of ({test2}) = ",XOR_GATE(test2))
print(f"XOR gate of ({test3}) = ",XOR_GATE(test3))
print(f"XOR gate of ({test4}) = ",XOR_GATE(test4))
