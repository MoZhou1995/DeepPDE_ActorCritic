# import numpy as np
# import tensorflow as tf
# import matplotlib.pyplot as plt
# import csv
print("hello")
# a = tf.constant([1.0, 2.0, 3.0, 4.0])
# index = tf.where(a>4.5)#shape num x 1
# num = tf.shape(index)[0]
# i=index[:,0]
# # j = tf.reshape(index,[1,2])
# b = tf.gather(a,index[:,0]) #for tensor shaped Num

# c = tf.constant([[1.0, 2.0], [3.0, 4.0],[5.0, 6.0], [7.0, 8.0]])
# d = tf.gather_nd(c,index) # for tensor shaped Num x d
# with open('ekn_10dTD3_T0.2_N100_R1.0_hist') as csv_file:
#     csv_reader = csv.reader(csv_file, delimiter=',')
#     line_count = 0
# import matplotlib.pyplot as plt
# import numpy as np
# import pandas as pd
# df = pd.read_csv (r'C:\Users\mozho\Mo Zhou\codes\DeepPDE\logs\ekn_10dTD3_T0.2_N100_R1.0_hist.csv')
# df = np.array(df)
# x = df[:,0:10]
# y = df[:,10]
# true_y = df[:,11]
# r = np.sum(x**2,1)**0.5
# f = plt.figure()
# ax = f.add_subplot(111)
# ax.plot(r,y,'ro',label='value_r_V')
# ax.plot(r,true_y,'bo', label='true_value')
# plt.legend()
# y = pd.Series(y)
# true_y= pd.Series(true_y)
# yy = pd.DataFrame({'y':y, 'true_y':true_y,})
# yy.plot.kde()
# yy.plot.hist(bins=30)
# fig, ax = plt.subplots()
# y.plot.kde(ax=ax, legend=False, title='Histogram')
