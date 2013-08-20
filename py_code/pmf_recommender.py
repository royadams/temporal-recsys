# The MIT License (MIT)
#
# Copyright (c) 2013 Roy Adams
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

import numpy as np
import pickle as pk
import scipy.sparse as sp
from scipy.stats import kendalltau as kt

#############################################################################
# SET THESE
#############################################################################
model_folder = <path_to_model_folder>
output_folder = <path_to_output_folder>
data_folder = <path_to_data_folder>
#############################################################################

class pmf:
	def __init__(self,fn):
		self.test_fn = ""
		self.fn = fn
		self.file_base = output_folder+fn+"/"+fn
		self.rank = 0
		self.ks = []
		self.nuff = 0
		self.nvff = 0
		self.nk = 0
		self.U = []
		self.u_cols = []
		self.Uff = []
		self.uff_cols = []
		self.maxes = []
		self.V = []
		self.d0_feats = []
		self.d1_feats = []
		self.u_feats = []
		self.uff_feats = []
		self.uff_ranges = []
		self.vff_ranges = []
		self.d0_ranges = []
		self.d1_ranges = []
		##############################################################
		# Read Model File
		##############################################################
		print("*** Reading Model File ***")
		with open(self.file_base+".model") as f:
			for l in f:
				[lab,val] = l.split("=")
				if lab == "rank":
					self.rank = int(val)
				elif lab ==  "nu":
					self.nu = int(val)
				elif lab == "ni":
					self.ni = int(val)
				elif lab == "nk":
					self.nk = int(val)
				elif lab == "nuff":
					self.nuff = int(val)
				elif lab == "nvff":
					self.nvff = int(val)
				elif lab == "nb":
					self.nb = int(val)
				elif lab == "mu":
					self.mu = float(val)
				elif lab == "uff_feats":
					self.uff_feats = val.split(";")[:-1]
				elif lab == "vff_feats":
					self.vff_feats = val.split(";")[:-1]
				elif lab == "d0_feats":
					self.d0_feats = val.split(";")[:-1]
				elif lab == "d1_feats":
					self.d1_feats = val.split(";")[:-1]
				elif lab == "ranks":
					self.ks = map(int,val.split(";")[:-1])
				elif lab == "u_features":
					self.u_feats = val.split(";")[:-1]
				elif lab == "w_feats":
					self.w_feats = val.split(";")[:-1]
				elif lab == "nw":
					self.nw = int(val)
				elif lab == "uff_ranges":
					self.uff_ranges = map(int,val.split(";")[:-1])
				elif lab == "vff_ranges":
					self.vff_ranges = map(int,val.split(";")[:-1])
				elif lab == "d0_ranges":
					self.d0_ranges = map(int,val.split(";")[:-1])
				elif lab == "d1_ranges":
					self.d1_ranges = map(int,val.split(";")[:-1])
				else:
					print("Invalid model file.")
					print(val,lab)
					return
		
		
		print("*** Loading Model ***")
		##############################################################
		# Load B
		##############################################################
		if(self.nb > 0):
			self.B = []
			self.d0s = []
			self.d1s = []
			for i in range(self.nb):
				self.B.append(np.fromfile("%s.B.%d"%(self.file_base,i),np.float64))
				self.B[i] = self.B[i].reshape((self.d0_ranges[i],self.d1_ranges[i]))
		
		
		##############################################################
		# Load W
		##############################################################
		if self.nw > 0:
			self.W = np.fromfile("%s.W"%(self.file_base,i),np.float64)
		
		##############################################################
		# Load U and V
		##############################################################
		if(self.rank > 0):
			self.U = np.fromfile("%s.U"%self.file_base,np.float64).reshape((self.nu,self.rank))
			self.V = np.fromfile("%s.V"%self.file_base,np.float64).reshape((self.ni,self.rank))
		
		##############################################################
		# Load UFF
		##############################################################
		for i in range(self.nuff):
			self.maxes.append(np.fromfile("%s.UFF.%d.maxes"%(self.file_base,i),np.int32))
			uff_raw = np.fromfile("%s.UFF.%d"%(self.file_base,i),np.float64)
			uff_raw[uff_raw < -1000] = 0.0
			users = []
			times = []
			inds = []
			m_prev = 0
			for u,m in zip(range(len(self.maxes[i])),self.maxes[i]):
				times += range(m+1)
				users += (m+1)*[u]
			inds = np.arange(len(times))
			self.Uff.append([])
			for j in range(self.rank):
				# T = sp.coo_matrix((uff_raw[(np.array(times)+j)],(users,times))).tocsr()
				self.Uff[i].append(sp.coo_matrix((uff_raw[((inds*self.rank)+j)],(users,times))).tocsr())
				self.Uff[i][j].sort_indices()
				self.Uff[i][j].eliminate_zeros()
		
		##############################################################
		# Load split U
		##############################################################
		for i in range(self.nk):
			self.maxes.append(np.fromfile("%s.U.%d.maxes"%(self.file_base,i),np.int32))
			u_raw = np.fromfile("%s.U.%d"%(self.file_base,i),np.float64)
			u_raw[u_raw < -1000] = 0.0
			users = []
			times = []
			inds = []
			m_prev = 0
			for u,m in zip(range(len(self.maxes[i])),self.maxes[i]):
				times += range(m+1)
				users += (m+1)*[u]
			inds = np.arange(len(times))
			self.U.append([])
			for j in range(self.ks[i]):
				self.U[i].append(sp.coo_matrix((u_raw[((inds*self.ks[i])+j)],(users,times))).tocsr())
				self.U[i][j].sort_indices()
				self.U[i][j].eliminate_zeros()
			self.V.append(np.fromfile("%s.V.%d"%(self.file_base,i),np.float64).reshape((self.ni,self.ks[i])))

	#########################################################################
	# Load a test set and set the data file columns for the various model 
	# components
	#########################################################################
	def load_test_set(self,fn):
		with open(data_folder+fn) as f:
			self.feats = f.readline()[:-1].split(",") + ["const"]
		try:
			self.d0s = [self.feats.index(feat) for feat in self.d0_feats]
			self.d1s = [self.feats.index(feat) for feat in self.d1_feats]
			self.u_cols = [self.feats.index(feat) for feat in self.u_feats]
			self.uff_cols = [self.feats.index(feat) for feat in self.uff_feats]
			self.w_cols = [self.feats.index(feat) for feat in self.w_feats]
		except:
			print "Test file does not match model file."
			raise
			
		if self.test_fn != fn:
			self.test_fn = fn
			test_data = np.loadtxt(data_folder+fn,dtype=np.int32,delimiter=",",skiprows=2)
			self.test_data = np.hstack((test_data,np.zeros((test_data.shape[0],1),dtype=np.int32)))
	
	#########################################################################
	# Calculate RMSE on specified test file
	#########################################################################
	def rmse(self,test_fn,test_rows = []):

		try:
			self.load_test_set(test_fn)
		except:
			raise
			
		if test_rows != []:
			test_data = self.test_data[test_rows]
		else:
			test_data = self.test_data
			
		tot_sq_err = 0.0
		for ex in test_data:
			u = ex[0]
			i = ex[1]
			pred = self.mu
			# if u % 1000 == 0:
				# print u
			for b in range(self.nb):
				if self.B[b][ex[self.d0s[b]]][ex[self.d1s[b]]] > -1000:
					pred += self.B[b][ex[self.d0s[b]]][ex[self.d1s[b]]]
			for w in range(self.nw):
				pred += self.W[w]*ex[self.w_cols[w]]
			if self.rank > 0:
				u_buff = np.copy(self.U[u])
				for ff in range(self.nuff):
					ind = np.where(self.Uff[ff][0].indices[self.Uff[ff][0].indptr[u]:self.Uff[ff][0].indptr[u+1]] == ex[self.uff_cols[ff]])[0]
					if ind.shape[0] == 1:
						for k in range(self.rank):
							u_buff[k] += self.Uff[ff][k].data[self.Uff[ff][0].indptr[u]+ind[0]]
						
				pred += np.dot(u_buff,self.V[i])
			for f in range(self.nk):
				ind = np.where(self.U[f][0].indices[self.U[f][0].indptr[u]:self.U[f][0].indptr[u+1]] == ex[self.u_cols[f]])[0]
				if ind.shape[0] == 1:
					for k in range(self.ks[f]):
						pred += self.U[f][k].data[self.U[f][0].indptr[u]+ind[0]]*self.V[f][i,k]
			if pred > 5:
				pred = 5
			if pred < 1:
				pred = 1
			tot_sq_err += (pred - ex[2])**2
		return np.sqrt(tot_sq_err/test_data.shape[0])
		
	#########################################################################
	# Calculate predictions on specified test file
	#########################################################################
	def preds(self,test_fn,test_rows = []):

		try:
			self.load_test_set(test_fn)
		except:
			raise
			
		if test_rows != []:
			test_data = self.test_data[test_rows]
		else:
			test_data = self.test_data
			
		preds = np.zeros(test_data.shape[0])
			
		ind = 0
		for ex in test_data:
			u = ex[0]
			i = ex[1]
			pred = self.mu
			# if u % 1000 == 0:
				# print u
			for b in range(self.nb):
				if self.B[b][ex[self.d0s[b]]][ex[self.d1s[b]]] > -1000:
					pred += self.B[b][ex[self.d0s[b]]][ex[self.d1s[b]]]
			for w in range(self.nw):
				pred += self.W[w]*ex[self.w_cols[w]]
			if self.rank > 0:
				u_buff = np.copy(self.U[u])
				for ff in range(self.nuff):
					ind = np.where(self.Uff[ff][0].indices[self.Uff[ff][0].indptr[u]:self.Uff[ff][0].indptr[u+1]] == ex[self.uff_cols[ff]])[0]
					if ind.shape[0] == 1:
						for k in range(self.rank):
							u_buff[k] += self.Uff[ff][k].data[self.Uff[ff][0].indptr[u]+ind[0]]
						
				pred += np.dot(u_buff,self.V[i])
			for f in range(self.nk):
				ind = np.where(self.U[f][0].indices[self.U[f][0].indptr[u]:self.U[f][0].indptr[u+1]] == ex[self.u_cols[f]])[0]
				if ind.shape[0] == 1:
					for k in range(self.ks[f]):
						pred += self.U[f][k].data[self.U[f][0].indptr[u]+ind[0]]*self.V[f][i,k]
			if pred > 5:
				pred = 5
			if pred < 1:
				pred = 1
			preds[ind] = pred
			ind+=1
		return preds
	
	#########################################################################
	# Calculate Kendall's Tau-B on specified test file
	#########################################################################
	def rank_loss_kt(self,test_fn,test_rows = []):
		preds = self.preds(test_fn)
		
		taus = []
		for u in range(self.nu):
			u_mask = self.test_data[:,0] == u
			u_ratings = self.test_data[u_mask,:]
			u_preds = preds[u_mask]
			nrat = u_ratings.shape[0]
			if nrat == 1:
				print "Insufficient ratings per user"
				return 0
			elif nrat == 0:
				continue
			elif np.all(u_ratings[:,2] == u_ratings[0,2]):
				continue
			elif np.all(u_preds == u_preds[0]):
				taus.append(0)
			
			taus.append(kt(u_preds,u_ratings[:,2])[0])
			
		print(len(taus))
		tau = np.mean(np.array(taus))
		
		return tau
				
	#########################################################################
	# Use nearest neighbor interpolation to fill in parameter values,
	# indexed by smoothed_feature, that were never trained.
	#########################################################################
	def rmse_nn_int(self,test_fn,smoothed_feature,test_rows = []):
		try:
			self.load_test_set(test_fn)
		except:
			raise
			
		if test_rows != []:
			test_data = self.test_data[test_rows]
		else:
			test_data = self.test_data
			
		if smoothed_feature in self.d0_feats:
			smoothed_feature_range = B[self.d0_feats.index(smoothed_feature)].shape[0]
		if smoothed_feature in self.d1_feats:
			smoothed_feature_range = self.B[self.d1_feats.index(smoothed_feature)].shape[1]
		if smoothed_feature in self.uff_feats:
			smoothed_feature_range = self.Uff[self.uff_feats.index(smoothed_feature)][0].shape[1]
		if smoothed_feature in self.u_feats:
			smoothed_feature_range = self.U[self.u_feats.index(smoothed_feature)][0].shape[1]
		
		tot_sq_err = 0.0
		for ex in test_data:
			u = ex[0]
			item = ex[1]
			pred = self.mu
			for b in range(self.nb):
				if self.B[b][ex[self.d0s[b]]][ex[self.d1s[b]]] > -1000:
					pred += self.B[b][ex[self.d0s[b]]][ex[self.d1s[b]]]
				else:
					t = int(ex[self.d1s[b]])
					t_nn_gt = -1
					t_nn_lt = -1
					for i in range(t,smoothed_feature_range):
						if self.B[b][ex[self.d0s[b]]][i] > -1000:
							t_nn_gt = i
							break
					for i in reversed(range(t)):
						if self.B[b][ex[self.d0s[b]]][i] > -1000:
							t_nn_lt = i
							break
					if abs(t - t_nn_gt) < abs(t - t_nn_lt):
						pred += self.B[b][ex[self.d0s[b]]][t_nn_gt]
					else:
						pred += self.B[b][ex[self.d0s[b]]][t_nn_lt]
			if self.rank > 0:
				u_buff = np.copy(self.U[u])
				for ff in range(self.nuff):
					ind = np.where(self.Uff[ff][0].indices[self.Uff[ff][0].indptr[u]:self.Uff[ff][0].indptr[u+1]] == ex[self.uff_cols[ff]])[0]
					if ind.shape[0] == 1:
						for k in range(self.rank):
							u_buff[k] += self.Uff[ff][k].data[self.Uff[ff][0].indptr[u]+ind[0]]
					else:
						i = self.Uff[ff][0].indptr[u+1]-self.Uff[ff][0].indptr[u]-1
						for t in reversed(self.Uff[ff][0].indices[self.Uff[ff][0].indptr[u]:self.Uff[ff][0].indptr[u+1]]):
							if t < ex[self.uff_cols[ff]]:
								break
							i -= 1
						if i == -1:
							for k in range(self.rank):
								u_buff[k] += self.Uff[ff][0].data[self.Uff[ff][0].indptr[u]]
						elif abs(ex[self.uff_cols[ff]] - self.Uff[ff][0].indices[i]) > abs((ex[self.uff_cols[ff]] - self.Uff[ff][0].indices[i+1])):
							for k in range(self.rank):
								u_buff[k] += self.Uff[ff][0].data[self.Uff[ff][0].indptr[u]+i+1]
						else:
							for k in range(self.rank):
								u_buff[k] += self.Uff[ff][0].data[self.Uff[ff][0].indptr[u]+i]
				pred += np.dot(u_buff,self.V[item])
			for f in range(self.nk):
				ind = np.where(self.U[f][0].indices[self.U[f][0].indptr[u]:self.U[f][0].indptr[u+1]] == ex[self.u_cols[f]])[0]
				if ind.shape[0] == 1:
					for k in range(self.ks[f]):
						pred += self.U[f][k].data[self.U[f][0].indptr[u]+ind[0]]*self.V[f][i,k]
				else:
					i = self.U[f][0].indptr[u+1]-self.U[f][0].indptr[u]-1
					for t in reversed(self.U[f][0].indices[self.U[f][0].indptr[u]:self.U[f][0].indptr[u+1]]):
						if t < ex[self.u_cols[f]]:
							break
						i -= 1
					if i == -1:
						for k in range(self.ks[f]):
							pred += self.U[f][0].data[self.U[f][0].indptr[u]]* self.V[i][item,k]
					elif abs(ex[self.u_cols[f]] - self.U[f][0].indices[i]) > abs((ex[self.u_cols[f]] - self.U[f][0].indices[i+1])):
						for k in range(self.ks[f]):
							pred += self.U[f][0].data[self.U[f][0].indptr[u]+i+1]* self.V[i][item,k]
					else:
						for k in range(self.ks[f]):
							pred += self.U[f][0].data[self.U[f][0].indptr[u]+i]* self.V[i][item,k]
			if pred > 100:
				pred = 100
			if pred < 0:
				pred = 0
			tot_sq_err += (pred - ex[2])**2
		return np.sqrt(tot_sq_err/test_data.shape[0])
		
	#########################################################################
	# Use squared exponential interpolation to fill in parameter values,
	# indexed by smoothed_feature, that were never trained.
	#########################################################################
	def rmse_sq_exp_int(self,test_fn,smoothed_feature,b_inv,test_rows = []):
		try:
			self.load_test_set(test_fn)
		except:
			raise
			
		if test_rows != []:
			test_data = self.test_data[test_rows]
		else:
			test_data = self.test_data
		
		if smoothed_feature in self.d0_feats:
			smoothed_feature_range = B[self.d0_feats.index(smoothed_feature)].shape[0]
		if smoothed_feature in self.d1_feats:
			smoothed_feature_range = self.B[self.d1_feats.index(smoothed_feature)].shape[1]
		if smoothed_feature in self.uff_feats:
			smoothed_feature_range = self.Uff[self.uff_feats.index(smoothed_feature)][0].shape[1]
		if smoothed_feature in self.u_feats:
			smoothed_feature_range = self.U[self.u_feats.index(smoothed_feature)][0].shape[1]
		
		Sigma = np.ones((smoothed_feature_range,smoothed_feature_range))
		for t0 in range(smoothed_feature_range):
			for t1 in range(smoothed_feature_range):
				Sigma[t0,t1] = np.exp(-((t0-t1)**2)/b_inv)
		Sigma_inv = np.linalg.inv(Sigma)
		tot_sq_err = 0.0
		for ex in test_data:
			u = ex[0]
			if u % 1000 == 0:
				print u
			i = ex[1]
			pred = self.mu
			for b in range(self.nb):
				if self.B[b][ex[self.d0s[b]]][ex[self.d1s[b]]] > -1000:
					pred += self.B[b][ex[self.d0s[b]]][ex[self.d1s[b]]]
				else:
					seen = self.B[b][ex[self.d0s[b]]] > -1000
					pred += np.dot(np.dot(Sigma[ex[self.d1s[b]],seen],Sigma_inv[seen][:,seen]),self.B[b][ex[self.d0s[b]]][seen])
			if self.rank > 0:
				u_buff = np.copy(self.U[u])
				for ff in range(self.nuff):
					ind = np.where(self.Uff[ff][0].indices[self.Uff[ff][0].indptr[u]:self.Uff[ff][0].indptr[u+1]] == ex[self.uff_cols[ff]])[0]
					if ind.shape[0] == 1:
						for k in range(self.rank):
							u_buff[k] += self.Uff[ff][k].data[self.Uff[ff][0].indptr[u]+ind[0]]
					else:
						seen = self.Uff[ff][0].indices[self.Uff[ff][0].indptr[u]:self.Uff[ff][0].indptr[u+1]]
						offset = self.Uff[ff][0].indptr[u]
						for k in range(self.rank):
							u_buff[k] += np.dot(np.dot(Sigma[ex[self.uff_cols[ff]],seen],Sigma_inv[seen][:,seen]),self.Uff[ff][k].data[self.Uff[ff][0].indptr[u]:self.Uff[ff][0].indptr[u+1]])
				pred += np.dot(u_buff,self.V[i])
			for f in range(self.nk):
				ind = np.where(self.U[f][0].indices[self.U[f][0].indptr[u]:self.U[f][0].indptr[u+1]] == ex[self.u_cols[f]])[0]
				if ind.shape[0] == 1:
					for k in range(self.ks[f]):
						pred += self.U[f][k].data[self.U[f][0].indptr[u]+ind[0]]*self.V[f][i,k]
				else:
					seen = self.U[f][0].indices[self.U[f][0].indptr[u]:self.U[f][0].indptr[u+1]]
					offset = self.U[f][0].indptr[u]
					for k in range(self.ks[f]):
						pred += np.dot(np.dot(Sigma[ex[self.u_cols[f]],seen],Sigma_inv[seen][:,seen]),self.U[f][k].data[self.U[f][0].indptr[u]:self.U[f][0].indptr[u+1]])*self.V[f][i,k]
			if pred > 100:
				pred = 100
			if pred < 0:
				pred = 0
			tot_sq_err += (pred - ex[2])**2
		return np.sqrt(tot_sq_err/test_data.shape[0])
		
	#########################################################################
	# Use squared exponential (GP) smoothing to smooth parameter values across
	# the smoothed_parameter axis
	#########################################################################
	def rmse_sq_exp_smoothing(self,test_fn,smoothed_feature,b_inv,s2,test_rows = []):
		try:
			self.load_test_set(test_fn)
		except:
			raise
			
		if test_rows != []:
			test_data = self.test_data[test_rows]
		else:
			test_data = self.test_data
		
		if smoothed_feature in self.d0_feats:
			smoothed_feature_range = B[self.d0_feats.index(smoothed_feature)].shape[0]
		if smoothed_feature in self.d1_feats:
			smoothed_feature_range = self.B[self.d1_feats.index(smoothed_feature)].shape[1]
		if smoothed_feature in self.uff_feats:
			smoothed_feature_range = self.Uff[self.uff_feats.index(smoothed_feature)][0].shape[1]
		if smoothed_feature in self.u_feats:
			smoothed_feature_range = self.U[self.u_feats.index(smoothed_feature)][0].shape[1]
		
		Sigma = np.ones((smoothed_feature_range,smoothed_feature_range))
		for t0 in range(smoothed_feature_range):
			for t1 in range(smoothed_feature_range):
				Sigma[t0,t1] = np.exp(-((t0-t1)**2)/b_inv)
		Sigma_inv = np.linalg.inv(Sigma+s2*np.identity(smoothed_feature_range))
		tot_sq_err = 0.0
		for ex in test_data:
			u = ex[0]
			i = ex[1]
			pred = self.mu
			for b in range(self.nb):
				if self.B[b][ex[self.d0s[b]]][ex[self.d1s[b]]] > -1000:
					pred += self.B[b][ex[self.d0s[b]]][ex[self.d1s[b]]]
				else:
					seen = self.B[b][ex[self.d0s[b]]] > -1000
					pred += np.dot(np.dot(Sigma[ex[self.d1s[b]],seen],Sigma_inv[seen][:,seen]),self.B[b][ex[self.d0s[b]]][seen])
			if self.rank > 0:
				u_buff = np.copy(self.U[u])
				for ff in range(self.nuff):
					ind = np.where(self.Uff[ff][0].indices[self.Uff[ff][0].indptr[u]:self.Uff[ff][0].indptr[u+1]] == ex[self.uff_cols[ff]])[0]
					if ind.shape[0] == 1:
						for k in range(self.rank):
							u_buff[k] += self.Uff[ff][k].data[self.Uff[ff][0].indptr[u]+ind[0]]
					else:
						seen = self.Uff[ff][0].indices[self.Uff[ff][0].indptr[u]:self.Uff[ff][0].indptr[u+1]]
						offset = self.Uff[ff][0].indptr[u]
						for k in range(self.rank):
							u_buff[k] += np.dot(np.dot(Sigma[ex[self.uff_cols[ff]],seen],Sigma_inv[seen][:,seen]),self.Uff[ff][k].data[offset+seen])
				pred += np.dot(u_buff,self.V[i])
			for f in range(self.nk):
				ind = np.where(self.U[f][0].indices[self.U[f][0].indptr[u]:self.U[f][0].indptr[u+1]] == ex[self.u_cols[f]])[0]
				if ind.shape[0] == 1:
					for k in range(self.ks[f]):
						pred += self.U[f][k].data[self.U[f][0].indptr[u]+ind[0]]*self.V[f][i,k]
				else:
					seen = self.U[f][0].indices[self.U[f][0].indptr[u]:self.U[f][0].indptr[u+1]]
					offset = self.U[f][0].indptr[u]
					for k in range(self.ks[f]):
						pred += np.dot(np.dot(Sigma[ex[self.u_cols[f]],seen],Sigma_inv[seen][:,seen]),self.U[f][k].data[offset+seen])*self.V[f][i,k]
			if pred > 100:
				pred = 100
			if pred < 0:
				pred = 0
			tot_sq_err += (pred - ex[2])**2
		return np.sqrt(tot_sq_err/test_data.shape[0])
		