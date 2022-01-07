import matplotlib.pyplot as plt
x=[i for i in range(10)]
print(x)
y=[3*i for i in range(10)]
print(y)

plt.xlabel('x-axis')
plt.ylabel('y-axis')
#plt.plot(x,y)

plt.scatter(x,y)
plt.show()