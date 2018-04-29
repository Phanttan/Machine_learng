import numpy as np
import pylab

# Initial values
x = np.linspace(-1, 1, 100)
print(x)
signal = 2 + x + 2*x*x
noise = np.random.normal(0, 0.5, 100)
y = signal + noise
pylab.plot(signal, 'b');
pylab.plot(y, 'g')
pylab.plot(noise,'r')
pylab.xlabel('x')
pylab.ylabel('y')
pylab.legend(["Without Noisy","With Noisy","Noise"],loc=2)
pylab.show()