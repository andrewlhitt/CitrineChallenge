import constraints
import sys
import numpy as np
import random 

def main(): 
	# Load cmd line arguments 
	input_file = sys.argv[1]
	output_file = sys.argv[2]
	n_results = int(sys.argv[3])

	# Set up Constraint object
	constraint = constraints.Constraint(input_file)
	example = constraint.get_example()
	n_dim = constraint.get_ndim()

	# points = list of points that have been checked against the constraints
	# labels = whether the point with the corresponding index in points obeys or violates the constraints
	points = list()
	labels = list()

	points.append(np.array(example))
	labels.append(constraint.apply(example))
	
	# generate points at the corners and classify them 
	for i in range(0,2**n_dim):
		str_corner = ('{0:0'+str(n_dim)+'b}').format(i)
		corner = [0.9999 if int(d) == 1 else 0.0001 for d in str_corner]
		try: 
			points.append(np.array(corner))
			labels.append(constraint.apply(corner))
		except ZeroDivisionError:
			print(corner, 'produces a divide by zero error')

	# good_points = points that obey the constraints
	# bad_points = points that violate the constraints 
	good_points = get_good_points(points,labels)
	bad_points = get_bad_points(points,labels)

	# select the example point as the first test_point
	test_point = good_points[0]

	# loop until 4000 (4*n_results) valid points have been identified (takes 5-10s)
	# for each loop, perform bisection search between a point (the "test_point") and the 2^n_dim corners 
	#   to find a valid point that is within tolerance (here 1e-5) of the boundary between valid and invalid points
	# the next test point is a random point selected from the unit hypercube and then confirmed to be a valid point
	# if the randomly selected point is not a valid point, it is regenerated (up to 20 times) 
	# if after 20 attempts, random selection over the unit hypercube has not produced a point
	#   the random selection process is then repeated over the n-dimensional bounding box that bounds all known valid points
	# random selection from points in the bounding box is repeated until it identifies a valid point 
	#   this will eventually (typically quite quickly, even on strongly-limited domains) produce a valid point
	# the process then repeats until 4000 valid points have been found
	# a vast majority of these valid points will exist within tolerance of the boundary
	while len(good_points) < 4*n_results:
		new_points = [edge_find_bisection(test_point,bad_point,1e-5,constraint) for bad_point in bad_points]
		new_labels = [True] * len(new_points)
		good_points = good_points + new_points
		points = points + new_points
		labels = labels + new_labels

		test_point = np.array([random.uniform(0,1) for i in range(0,n_dim)]) 

		attempts = 0
		while not constraint.apply(test_point) and attempts < 20:
			bad_points.append(test_point)
			points.append(test_point)
			labels.append(False)
			test_point = np.array([random.uniform(0,1) for i in range(0,n_dim)]) 
			attempts += 1 

		if attempts >= 20: 
			bounds_min = np.min(good_points,axis=0) 
			bounds_max = np.max(good_points,axis=0)
			test_point = np.array([random.uniform(bounds_min[i],bounds_max[i]) for i in range(0,n_dim)]) 
			while not constraint.apply(test_point):
				bad_points.append(test_point)
				points.append(test_point)
				labels.append(False)
				test_point = np.array([random.uniform(bounds_min[i],bounds_max[i]) for i in range(0,n_dim)]) 

		points.append(test_point)
		labels.append(True)
	

	# now that the edges have been mostly identified, it is time to fill in the interior
	# to fill in the interior, random points within the bounding box will be generated
	# this will continue until 1000 (n_results) new points have been generated 
	# this is to confirm that identified regions are simply connected (e.g. there are no holes in the interior of the identified space)
	interior_min = np.min(good_points,axis=0) 
	interior_max = np.max(good_points,axis=0) 
	interior_points = list()
	while len(interior_points) < n_results:
		test_point = np.array([random.uniform(interior_min[i],interior_max[i]) for i in range(0,n_dim)]) 
		if constraint.apply(test_point):
			interior_points.append(0)
			points.append(test_point)
			labels.append(True)
	
	# there should now be at least 5000 valid points to select from 
	# the best way to represent the valid region is to have both points near the boundary 
	#  as well as points in the interior of the region (far from any edges)
	# thus, the following algorithm identifies 500 points with smallest minimum distance to an invalid point (edges) 
	#  and 500 points with largest minimum distance to an invalid point (interior)
	good_points = get_good_points(points,labels)
	good_points_set = set(map(tuple,good_points))
	good_points_no_dups = list(map(list,good_points_set))
	
	bad_points = get_bad_points(points,labels)
	bad_points_set = set(map(tuple,bad_points))
	bad_points_no_dups = list(map(list,bad_points_set))
	bad_distances = np.array([min([get_distance(good_point,bad_point) for bad_point in bad_points_no_dups]) for good_point in good_points_no_dups])

	edge_count = int(0.5*n_results)
	interior_count = n_results-edge_count
	interior_indices = bad_distances.argsort()[-interior_count:]
	edge_indices = bad_distances.argsort()[::-1][:edge_count]

	output_interior = [good_points_no_dups[i] for i in interior_indices]
	output_edge = [good_points_no_dups[i] for i in edge_indices]

	output_all = output_interior + output_edge

	# write selected points to output file
	with open(output_file,'w') as f: 
		for i in range(0,len(output_all)):
			f.write(' '.join(str(x) for x in output_all[i])+'\n')

def edge_find_bisection(good_point,bad_point,minimum_distance,constraint):
	# input: two points of opposite labels 
	# output: a valid point arbitrarily close to the boundary 
	distance = get_distance(good_point,bad_point)
	output = good_point
	iterations = 0 
	while distance > minimum_distance:
		midpoint = (np.array(good_point)+np.array(bad_point))/2
		label = constraint.apply(midpoint)
		if label:
			good_point = midpoint
			distance = get_distance(good_point,bad_point)
			output = midpoint
		else: 
			bad_point = midpoint
			distance = get_distance(good_point,bad_point)
		#print(output,midpoint,distance)
		iterations += 1
	#print(iteration)
	return output
		#distance = get_distance(pointA,pointB) 

def get_distance(pointA,pointB):
	# input: two points (represented as lists or numpy vectors)
	# output: euclidean distance (squareroot of sum of squares of differences in components) 
	return np.sqrt(np.sum((np.subtract(pointA,pointB))**2))

def get_good_points(points,labels):
	return [points[i] for i in range(0,len(points)) if labels[i]]

def get_bad_points(points,labels):
	return [points[i] for i in range(0,len(points)) if not labels[i]]

def visualize(points,labels):
	# this function was used to help visualize the progress of the main algorithm 
	good_points = get_good_points(points,labels)
	bounds_min = np.min(points,axis=0) 
	bounds_max = np.max(points,axis=0)
	colors1 = ['red' if l == 0 else 'green' for l in labels]
	plt.scatter([p[0] for p in points],[p[1] for p in points],color=colors1,label='x[0],x[1]')
	plt.xlim((0,1))
	plt.ylim((0,1))
	plt.show()

if __name__ == '__main__':
	main()
