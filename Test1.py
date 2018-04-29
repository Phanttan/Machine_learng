import numpy as np
import pylab

pylab.close('all')                  # not working
# Initial values
x = np.linspace(-1, 1, 100)
signal = 2 + x + 2*x*x
noise = np.random.normal(0, 0.1, 100)
y = signal + noise
pylab.plot(signal, 'b');
pylab.plot(y,'g')
pylab.plot(noise, 'r')
pylab.xlabel('x')
pylab.ylabel('y')
pylab.legend(["Without Noisy", "With Noisy", "Noise"], loc=2)
# Training
x_train = x[0:80]
y_train = y[0:80]

# Model with Degree 1
pylab.figure()
degree = 2
X_train = np.column_stack([np.power(x_train,i) for i in range(degree)])
dot1 = np.dot(X_train.transpose(),X_train)
dot2 = np.dot(np.linalg.inv(dot1),X_train.transpose())
model = np.dot(dot2,y_train)
pylab.plot(x, y, 'g')
pylab.xlabel("x")
pylab.ylabel("y")
# Predicted
predicted = np.dot(model,[np.power(x, i) for i in range(degree)])
pylab.plot(x, predicted, 'r')
pylab.legend(["Actual", "Predicted"], loc=2)
train_po1 = np.dot(y[0:80]- predicted[0:80],y_train - predicted[0:80])
train_rmse1 = np.sqrt(np.sum(train_po1))
test_po1 = np.dot(y[80:]- predicted[80:],y[80:]- predicted[80:])
test_rmse1 = np.sqrt(np.sum(test_po1))
print("Train RMSE (Degree = 1) is", train_rmse1)
print("Test RMSE (Degree = 1) is ", test_rmse1)
pylab.show()








