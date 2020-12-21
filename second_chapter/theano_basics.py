import theano
import theano.tensor as T
from theano import function

if __name__ == "__main__":
  x = T.dscalar("x")
  y = T.dscalar("y")
  z = x + y
  f = function([x, y], z)

  print(x, y)
  print("="*50)
  print(f)