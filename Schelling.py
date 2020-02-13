import numpy as np
import cv2
from scipy import signal
from scipy import misc
from scipy.spatial import distance

# Initialize grid

grid = np.full((50, 50, 3), 255)
grid[..., 1] = np.zeros((50, 50))

# Simulation parameters

population = [0.5, 0.9]
threshold = [3, 4]
epochs = 500

# Indices in population and threshold 

i = 25

while not i in [0, 1]:
	i = int(input("Choose a proportion of occupied cells. Enter 0 for 0.5 or 1 for 0.9: "))

j = 25
while not j in [0, 1]:
	j = int(input("Choose a threshold for unsatisfaction. Enter 0 for 3 or 1 for 4: "))


# Random distribution of population

random = np.random.random_sample((50, 50))

red_selection = random.__lt__(population[i]/2)
grid[..., 0] = np.multiply(grid[..., 0], red_selection)

blue_selection1 = random.__gt__(population[i]/2)
grid[..., 2] = np.multiply(grid[..., 2], blue_selection1)
blue_selection2 = random.__lt__(population[i])
grid[..., 2] = np.multiply(grid[..., 2], blue_selection2)

# Function for returning the closest suitable point

def closest_point(point, list, matrix):	

	if np.shape(list)[0] == 0: 
		return point
	
	distances = []
	for item in list:
		distances.append(distance.chebyshev(point, item))
	
	minimum = min(distances)
	index = distances.index(minimum)
	move = list[distances.index(minimum)]

	while minimum == 1 and matrix[point[0]][point[1]] < (threshold[j]+1)*255 - 1:

		distances.pop(index)

		if np.shape(distances)[0] == 0:
			break

		minimum = min(distances)
		index = distances.index(minimum)
		move = list[index]
	
	if np.shape(distances)[0] == 0:
		return point
	else:
		return move	
			
# Convolution kernel

kernel = np.array([[1, 1, 1],
                   [1, 0, 1],
                   [1, 1, 1]])

# Function for evaluating satisfaction and moving 

def move(matrix):

	# Measure satisfactory pixels with convolution kernel

	red_satisfaction = signal.convolve2d(matrix[..., 0], kernel, mode = "same", boundary = "fill")
	red_satisfactory = red_satisfaction.__gt__(255*threshold[j] - 1)

	blue_satisfaction = signal.convolve2d(matrix[..., 2], kernel, mode = "same", boundary = "fill")
	blue_satisfactory = blue_satisfaction.__gt__(255*threshold[j] - 1)
	
	# Check for free pixels 

	no_red = matrix[..., 0].__lt__(255)
	no_blue = matrix[..., 2].__lt__(255)

	free_space = np.multiply(no_blue, no_red)

	# Combine for free pixels that would satisfy red or blue

	red_freeAndSatisfactory = np.multiply(free_space, red_satisfactory)
	red_freeAndSatisfactoryIndices = np.argwhere(red_freeAndSatisfactory == True).tolist()
	#np.random.shuffle(red_freeAndSatisfactoryIndices)

	blue_freeAndSatisfactory = np.multiply(free_space, blue_satisfactory)
	blue_freeAndSatisfactoryIndices = np.argwhere(blue_freeAndSatisfactory == True).tolist()
	#np.random.shuffle(blue_freeAndSatisfactoryIndices)

	# Check for pixels that are not currently satisfied

	red_unsatisfactory = red_satisfaction.__lt__(255*threshold[j])
	red = matrix[..., 0].__gt__(0)
	red_unsatisfied = np.multiply(red_unsatisfactory, red)
	red_unsatisfiedIndices = np.argwhere(red_unsatisfied == True).tolist()
	#np.random.shuffle(red_unsatisfiedIndices)

	blue_unsatisfactory = blue_satisfaction.__lt__(255*threshold[j])
	blue = matrix[..., 2].__gt__(2)
	blue_unsatisfied = np.multiply(blue_unsatisfactory, blue)
	blue_unsatisfiedIndices = np.argwhere(blue_unsatisfied == True).tolist()
	#np.random.shuffle(blue_unsatisfiedIndices)

	# Move unsatisfied pixels to random satisfactory location

	nb_moves = max(np.shape(blue_unsatisfiedIndices)[0], np.shape(red_unsatisfiedIndices)[0])

	k = 0

	while k < nb_moves:
		
		if k < np.shape(red_unsatisfiedIndices)[0] and k < np.shape(blue_unsatisfiedIndices)[0]:
			
			if len(red_freeAndSatisfactoryIndices) != 0:
	
				old_red = red_unsatisfiedIndices[k]
				matrix[..., 0][old_red[0]][old_red[1]] = 0
				
				new_red = closest_point(old_red, red_freeAndSatisfactoryIndices, red_satisfaction)
				matrix[..., 0][new_red[0]][new_red[1]] = 255

				if new_red in blue_freeAndSatisfactoryIndices:
					blue_freeAndSatisfactoryIndices.remove(new_red)
				
				if new_red != old_red:
					red_freeAndSatisfactoryIndices.remove(new_red)

			if len(blue_freeAndSatisfactoryIndices) != 0:

				old_blue = blue_unsatisfiedIndices[k]
				matrix[..., 2][old_blue[0]][old_blue[1]] = 0

				new_blue = closest_point(old_blue, blue_freeAndSatisfactoryIndices, blue_satisfaction)
				matrix[..., 2][new_blue[0]][new_blue[1]] = 255

				if new_blue in red_freeAndSatisfactoryIndices:
					red_freeAndSatisfactoryIndices.remove(new_blue)	
				
				if new_blue != old_blue:
					blue_freeAndSatisfactoryIndices.remove(new_blue)
		
		elif nb_moves == np.shape(red_unsatisfiedIndices)[0]:

			if len(red_freeAndSatisfactoryIndices) != 0:
	
				old_red = red_unsatisfiedIndices[k]
				matrix[..., 0][old_red[0]][old_red[1]] = 0

				new_red = closest_point(old_red, red_freeAndSatisfactoryIndices, red_satisfaction)
				matrix[..., 0][new_red[0]][new_red[1]] = 255

				if new_red in blue_freeAndSatisfactoryIndices:
					blue_freeAndSatisfactoryIndices.remove(new_red)

				if new_red != old_red:
					red_freeAndSatisfactoryIndices.remove(new_red)			

		else:

			if len(blue_freeAndSatisfactoryIndices) != 0:

				old_blue = blue_unsatisfiedIndices[k]
				matrix[..., 2][old_blue[0]][old_blue[1]] = 0

				new_blue = closest_point(old_blue, blue_freeAndSatisfactoryIndices, blue_satisfaction)
				matrix[..., 2][new_blue[0]][new_blue[1]] = 255

				if new_blue in red_freeAndSatisfactoryIndices:
					red_freeAndSatisfactoryIndices.remove(new_blue)	
				
				if new_blue != old_blue:
					blue_freeAndSatisfactoryIndices.remove(new_blue)
		
		k = k + 1

# Main loop 

for i in range(epochs):
	print("Iteration " + str(i +1))
	
	cv2.namedWindow("grid", cv2.WINDOW_NORMAL)
	cv2.resizeWindow("grid", 600, 600)
	cv2.imshow("grid", grid.astype(np.uint8))
	cv2.waitKey()
	
	move(grid)
	

#cv2.destroyAllWindows()

