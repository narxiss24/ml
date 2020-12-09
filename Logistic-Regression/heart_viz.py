import matplotlib.pyplot as plt
import pandas as pd

import heart_all

pd.set_option('display.max_rows', None)

df = heart_all.get_data()

plt.scatter(df['thalach'], df['hd1'])

plt.show()

