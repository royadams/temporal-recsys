import numpy as np
import scipy.sparse as sp

def partial_distance(x,xd,y,ndims):
	return np.sqrt(np.sum((x-y[xd])**2)) * ndims / len(x)

def kmeans_pd(data,k,max_iter,eps,verbose=False):
	if(not data.has_sorted_indices):
		data.sort_indices()
		
	[npoints,ndims] = data.shape
	iter = 	0
	delta = 99999999
	# Create and initialize cluster centers and old cluster centers
	assignments = np.zeros(npoints)
	centers = np.random.randn(k,ndims)
	while((not iter >= max_iter) and (not delta < eps)):
		# Assign points
		if(verbose):
			print("Iteration %g: Assigning Points"%iter)
		for p in range(npoints):
			min_dist = 999999999
			min_c = 0
			x = data.data[data.indptr[p]:data.indptr[p+1]]
			xd = data.indices[data.indptr[p]:data.indptr[p+1]]
			for c,cen in zip(range(k),centers):
				dist = np.sqrt(np.sum((x-cen[xd])**2)) * ndims / len(x)
				if dist < min_dist:
					min_c = c
					min_dist = dist
			assignments[p] = min_c
		# Recalculate cluster centers
		old_centers = centers
		counts = np.zeros((k,ndims))
		centers = np.zeros((k,ndims))
		if(verbose):
			print("Iteration %g: Recalculating Centers"%iter)
		for p in range(npoints):
			c = assignments[p]
			for d,v in zip(data.indices[data.indptr[p]:data.indptr[p+1]],data.data[data.indptr[p]:data.indptr[p+1]]):
				centers[c,d] += v
				counts[c,d] += 1
		counts[counts==0] = 1
		centers /= counts
		# Calculate delta 
		delta = np.sqrt(np.max(np.sum((old_centers - centers)**2,1))) 
		if(verbose):
			print("Iteration %g: delta = %g"%(iter,delta))
		iter += 1
	# Return cluster centers and assignments
	if(verbose and iter > max_iter):
			print("Max_iter reached. Exiting.")
	return centers,assignments