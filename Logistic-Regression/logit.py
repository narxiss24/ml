import matplotlib.pyplot as plt
from scipy.special import expit, logit

x_values = range(-6, 7)

log_values = [expit(x) for x in x_values]

lin_values = [logit(x) for x in log_values]

plt.plot(log_values, 'g-')

plt.show()
    


    

