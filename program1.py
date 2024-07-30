import matplotlib.pyplot as plt
hours = [10,9,2,15,10,16,11,16]
score = [95,80,10,50,45,98,38,93]
# Plotting the line chart
plt.plot(hours, score, marker='*', color='red', linestyle='-')
# Adding labels and title
plt.xlabel('Number of Hours Studied')
plt.ylabel('Score in Final Exam')
plt.title('Effect of Hours Studied on Exam Score')
# Displaying the plot
plt.grid(True)
plt.show()





########2nd program

import matplotlib.pyplot as plt
from collections import Counter
# Load the dataset
mtcars = pd.read_csv('mtcars.csv')  
# Plotting the histogram
histogram = Counter(min( mpg // 10 * 10, 90) for mpg in mtcars['mpg'])
plt.bar([x+5 for x in histogram.keys()],  # Shift bars right by 5
        histogram.values(),                 # Give each bar its correct height
        10,                                 # Give each bar a width of 8
        edgecolor= 'black')                # Black edges for each bar
plt.xticks([10 * i for i in range(6)])    # x-axis labels at 0, 10, ..., 50
# Adding labels and title
plt.xlabel('Miles per gallon (mpg)')
plt.ylabel('Frequency')
plt.title('Histogram of Miles per gallon (mpg)')
# Displaying the plot
plt.show()
