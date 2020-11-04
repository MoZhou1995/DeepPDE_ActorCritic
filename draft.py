import numpy as np
import tensorflow as tf

# a = tf.constant([1.0, 2.0, 3.0, 4.0])
# index = tf.where(a>4.5)#shape num x 1
# num = tf.shape(index)[0]
# i=index[:,0]
# # j = tf.reshape(index,[1,2])
# b = tf.gather(a,index[:,0]) #for tensor shaped Num

# c = tf.constant([[1.0, 2.0], [3.0, 4.0],[5.0, 6.0], [7.0, 8.0]])
# d = tf.gather_nd(c,index) # for tensor shaped Num x d

# if tf.shape(index)[0]>0:
#     print("yes")
# else:
#     print("no")
# rho = tf.constant([5.0, 6.0,])
# new_a = tf.tensor_scatter_nd_update(a, index, rho)

# def f1():
#     print("call f1")
#     return 1

# def f2():
#     print("call f2")
#     return 2

# def f(x,y):
#     return tf.cond(x > y, true_fn=f1, false_fn=f2)

# x = tf.constant([1.0, 2.0, 3.0, 4.0])
# y = tf.constant([4.0, 3.0, 2.0, 1.0])

# for j in range(4):
#     z_j = f(x[j], y[j])
  
#this one doesn't work
a = np.array([1.5762431408717797e-05])
b = np.array([-0.00023222736508783293])
c = np.array([-0.007020976166862999])
d = np.array([0.058056841271958225])
e = np.array([-0.014848036955137811])
'''
true solutions are
-19.015225949721957320
0.26426815937210765568
7.1067545228901608444
26.377169208196085873
'''

# this one works
# a = np.array([1.4811968472938404e-05])
# b = np.array([-0.0003781555488539054])
# c = np.array([-0.004220093814579385])
# d = np.array([0.09453888721347634])
# e = np.array([-0.07425197044134979])
# a+b+c+d+e = 0.03597136
p = (8*a*c - 3*(b**2)) / (8*(a**2))
q = (b**3 - 4*a*b*c + 8*(a**2)*d) / (8 * (a**3))
sign_q = np.sign(q)
Delta_0 = c**2 - 3*b*d + 12*a*e
Delta_1 = 2*(c**3) - 9*b*c*d + 27*(b**2)*e + 27*a*(d**2) - 72*a*c*e
Delta_2 = Delta_1**2 - 4*(Delta_0**3)
sign_Delta_2 = np.sign(Delta_2)
signal_Delta_2 = np.ceil((sign_Delta_2+1)/2)
# this is for sign_Delta >= 0
QQ = (Delta_1 + ( np.absolute(Delta_2) )**0.5 ) / 2 # the absolute value here is to make sure that no nan
Q = np.sign(QQ) * np.absolute(QQ)**(1/3) #I don't know why python cannot compute the cubic root of negative number
S_plus = 0.5 * np.absolute((Q + Delta_0/Q) / (3*a) - 2*p/3)**0.5
# This is for sign_Delta < 0
phi = np.arccos( np.minimum( np.absolute(Delta_1**2 / 4/ Delta_0**3)**0.5,1) )
S_minus = 0.5 * np.absolute(2 * (np.absolute(Delta_0)**0.5) * np.cos(phi/3) / (3*a) - 2*p/3)**0.5
S = signal_Delta_2 * S_plus + (1-signal_Delta_2) * S_minus
# either root1 or root3 is what we want.
root1 = 0.5 * (q/S - 4*(S**2) - 2*p)**0.5 - b/4/a - S
root2 = - 0.5 * (q/S - 4*(S**2) - 2*p)**0.5 - b/4/a - S
root3 = 0.5 * (-q/S - 4*(S**2) - 2*p)**0.5 - b/4/a + S
root4 = - 0.5 * (-q/S - 4*(S**2) - 2*p)**0.5 - b/4/a + S
root = np.array([root1[0],root2[0],root3[0],root4[0]])
temp = -4*(S**2) -2*p + np.absolute(q/S)
sqrt_rho = 0.5 * np.absolute(temp)**0.5 - b/4/a - sign_q*S
result = a*(root**4) + b*(root**3) + c*(root**2) + d*root+e
#D = 64*(a**3)*e - 16*(a**2)*(c**2) + 16*a*(b**2)*c - 16*(a**2)*b*d - 3*(b**4)

# phi2 = 2*np.pi - phi
# phi3 = 2*np.pi + phi
# S2 = 0.5 * (2 * (np.absolute(Delta_0)**0.5) * np.cos(phi2/3) / (3*a) - 2*p/3)**0.5
# S3 = 0.5 * (2 * (np.absolute(Delta_0)**0.5) * np.cos(phi3/3) / (3*a) - 2*p/3)**0.5
# temp2 = -4*(S2**2) -2*p + np.absolute(q/S2)
# temp3 = -4*(S3**2) -2*p + np.absolute(q/S3)
# sqrt_rho2 = 0.5 * np.absolute(temp2)**0.5 - b/4/a - sign_q*S2
# sqrt_rho3 = 0.5 * np.absolute(temp3)**0.5 - b/4/a - sign_q*S3
# result2 = a*(sqrt_rho2**4) + b*(sqrt_rho2**3) + c*(sqrt_rho2**2) + d*sqrt_rho2+e
# result3 = a*(sqrt_rho3**4) + b*(sqrt_rho3**3) + c*(sqrt_rho3**2) + d*sqrt_rho3+e
r2=-19.015225949721957320
r1=0.26426815937210765568
r4=7.1067545228901608444
r3=26.377169208196085873
S_true = (r3+r4-r1-r2)/4



