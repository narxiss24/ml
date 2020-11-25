import matplotlib.pyplot as plt
import numpy as np

x = range(0, 9)
y = (25, 33, 41, 53, 59, 70, 78, 86, 96)

plt.scatter(x,y)

z = np.polyfit(x, y, 500)
p = np.poly1d(z)

plt.plot(x, p(x), 'g-')

plt.show()
