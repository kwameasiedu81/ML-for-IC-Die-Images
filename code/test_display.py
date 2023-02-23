import matplotlib
import matplotlib.pyplot as plt
plt.style.use('ggplot')
matplotlib.use( 'tkagg' )
x = [1, 5, 1.5, 4]
y = [9, 1.8, 8, 11]
plt.scatter(x,y)
plt.show()