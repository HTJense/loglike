import numpy as np
import sys
import re
import itertools
from scipy import linalg
from scipy.io import FortranFile

class Likelihood:
	def __init__(self):
		self.win_func = None
		self.covmat = None
		self.cells = None
		self.b_dat = None
		self.win_ells = None
		self.b_ell = None
		self.ells = None
		
		# Total number of bins + number of bins per cross spectrum
		self.nbintt = 52
		self.nbinte = 52
		self.nbinee = 52
		
		self.nspectt = 3
		self.nspecte = 4
		self.nspecee = 3
		self.freqs = [ 95, 150 ]
		
		# maximum ell for windows and for cross spectra
		self.lmax_win = 7925
		self.tt_lmax = 6000
		
		# amount of low-ell bins to ignore per cross spectrum
		self.b0 = 5
		self.b1 = 0
		self.b2 = 0
		
		# calibration terms
		self.ct1 = 1.0
		self.ct2 = 1.0
		
		# leakage terms
		self.a1 = 0.0
		self.a2 = 0.0
		self.l98 = None
		self.l150 = None
		
		# Whether or not to use TT, EE and/or TE (or all)
		self.enable_tt = True
		self.enable_te = True
		self.enable_ee = True
	
	def load_plaintext(self, spec_filename, cov_filename, bbl_filename, data_dir = ''):
		self.win_func = np.loadtxt(data_dir + bbl_filename)[:self.nbin,:self.lmax_win]
		self.b_dat = np.loadtxt(data_dir + spec_filename)[:self.nbin]
		
		self.covmat = np.loadtxt(data_dir + cov_filename, dtype = float)[:self.nbin,:self.nbin] #.reshape((self.nbin, self.nbin))
		for i in range(self.nbin):
			for j in range(i, self.nbin):
				self.covmat[i,j] = self.covmat[j,i]

		self.cull_covmat()
	
	def load_sacc(self, sacc_filename, data_dir = '', xp_name = 'LAT'):
		try:
			import sacc
		except ImportError as e:
			print('Failed to load data from a SACC file: failed to import sacc.\n{}'.format(str(e)))
			return
		
		saccfile = sacc.Sacc.load_fits(data_dir + sacc_filename)
		
		# Find how many TT+TE+EE spectra are given in this file.
		self.nspectt = len(saccfile.get_tracer_combinations('cl_00'))
		self.nspecte = len(saccfile.get_tracer_combinations('cl_0e'))
		self.nspecee = len(saccfile.get_tracer_combinations('cl_ee'))
		
		# Check if the numbers add up, we expect N(N+1)/2 for TT/EE and N^2 for TE.
		n = int(np.sqrt(self.nspecte))
		ntt = int(n * (n + 1) / 2)
		nte = int(n * n)
		
		if self.nspectt != ntt or self.nspecte != nte or self.nspecee != ntt:
			raise ValueError('Incorrect number of spectra found: expected {}+{}+{} but found {}+{}+{} (TT+TE+EE).'.format(ntt, nte, ntt, self.nspectt, self.nspecte, self.nspecee))
		
		# List all used frequencies.
		# casting a list to a set back to a list to filter out only unique values.
		self.freqs = sorted(list(set([ int(re.search('{}_([0-9]+)_'.format(xp_name), x).groups()[0]) for x,_ in saccfile.get_tracer_combinations('cl_00') ])))
		
		# We check how large our window function + covmat should be
		_, tmp_cells = saccfile.get_ell_cl('cl_00', *saccfile.get_tracer_combinations('cl_00')[0], return_cov = False)
		self.nbintt = tmp_cells.shape[0]
		
		_, tmp_cells = saccfile.get_ell_cl('cl_0e', *saccfile.get_tracer_combinations('cl_0e')[0], return_cov = False)
		self.nbinte = tmp_cells.shape[0]
		
		_, tmp_cells = saccfile.get_ell_cl('cl_ee', *saccfile.get_tracer_combinations('cl_ee')[0], return_cov = False)
		self.nbinee = tmp_cells.shape[0]
		
		indices = saccfile.indices('cl_00', saccfile.get_tracer_combinations('cl_00')[0])
		win_func = saccfile.get_bandpower_windows(indices)
		self.lmax_win = win_func.values.shape[0]+1
		self.tt_lmax = np.nanmax(win_func.values)
		self.win_ells = win_func.weight.T @ win_func.values
		
		# We prepare our covariance matrix and window function
		self.covmat = np.zeros((self.nbin, self.nbin))
		self.b_dat = np.zeros((self.nbin))
		self.b_ell = np.zeros((self.nbin))
		self.win_func = np.zeros((self.nbin, self.shape))
		
		# now we jump along the diagonal of the covmat and include matrix blocks from the read-in covmat.
		j = 0
		for i, (f1, f2) in enumerate(itertools.product(self.freqs, self.freqs)):
			idx = (i % len(self.freqs), i // len(self.freqs))
			# We can check whether F1xF2 should be included in the covmat (note F1xF2 should not be in the covmat if F2xF1 is already in there and F1 != F2)
			# by checking whether the index is in the lower triangle of a matrix (i.e. we exclude the upper triangle).
			if idx in zip(*np.tril_indices(len(self.freqs))):
				eltt, cltt, covtt, ind_tt = saccfile.get_ell_cl('cl_00', '{}_{}_s0'.format(xp_name, f1), '{}_{}_s0'.format(xp_name, f2), return_cov = True, return_ind = True)
				win_tt = saccfile.get_bandpower_windows(ind_tt)
				
				n0 = j * self.nbintt
				self.b_dat[n0:n0+self.nbintt] = cltt
				self.b_ell[n0:n0+self.nbintt] = eltt
				self.covmat[n0:n0+self.nbintt,n0:n0+self.nbintt] = covtt[:,:]
				self.win_func[n0:n0+self.nbintt,:] = win_tt.weight[:,:].T
				
				elee, clee, covee, ind_ee = saccfile.get_ell_cl('cl_ee', '{}_{}_s2'.format(xp_name, f1), '{}_{}_s2'.format(xp_name, f2), return_cov = True, return_ind = True)
				win_ee = saccfile.get_bandpower_windows(ind_ee)
				
				n0 = self.nspectt * self.nbintt + self.nspecte * self.nbinte + j * self.nbinee
				self.b_dat[n0:n0+self.nbinee] = clee
				self.b_ell[n0:n0+self.nbinee] = elee
				self.covmat[n0:n0+self.nbinee,n0:n0+self.nbinee] = covee[:,:]
				self.win_func[n0:n0+self.nbinee,:] = win_ee.weight[:,:].T
				
				j += 1
			
			# TE can be done in all cases, because F1xF2 != F2xF1
			n0 = self.nspectt * self.nbintt + i * self.nbinte
			elte, clte, covte, ind_te = saccfile.get_ell_cl('cl_0e', '{}_{}_s0'.format(xp_name, f1), '{}_{}_s2'.format(xp_name, f2), return_cov = True, return_ind = True)
			
			if covte.shape == (0,0):
				# Sometimes the tracers are in the other order (they always store them such that F1 < F2), so we gotta reverse the order
				elte, clte, covte, ind_te = saccfile.get_ell_cl('cl_0e', '{}_{}_s2'.format(xp_name, f2), '{}_{}_s0'.format(xp_name, f1), return_cov = True, return_ind = True)
			
			win_te = saccfile.get_bandpower_windows(ind_te)
			self.b_dat[n0:n0+self.nbinte] = clte
			self.b_ell[n0:n0+self.nbinte] = elte
			self.covmat[n0:n0+self.nbinte,n0:n0+self.nbinte] = covte[:,:]
			self.win_func[n0:n0+self.nbinte,:] = win_te.weight[:,:].T
		
		self.cull_covmat()
	
	def load_cells(self, cl_filename, data_dir = '', c0_ells = False):
		self.cells = np.loadtxt(data_dir + cl_filename)[:self.shape,:]
		self.ells = np.arange(2, self.cells.shape[0] + 2)
		
		# set c0_ells to TRUE if the first column of the file contains the ells.
		if c0_ells:
			self.ells = self.cells[:,0]
			self.cells = self.cells[:,1:]
		
		self.tt_lmax = int(self.ells[-1])
	
	def load_cells_camb(self, lmax, cambparams, initpower = None, lens_potential_accuracy = 0):
		try:
			import camb
		except ImportError as e:
			print('Failed to load Cells from CAMB: failed to import camb.\n{}'.format(str(e)))
			return
		
		self.tt_lmax = lmax
		
		pars = camb.CAMBparams(**cambparams)
		if not initpower is None: pars.InitPower.set_params(**initpower)
		pars.set_for_lmax(self.tt_lmax, lens_potential_accuracy = lens_potential_accuracy)
		res = camb.get_results(pars)
		powers = res.get_cmb_power_spectra(pars, CMB_unit = 'muK')['total']
		
		self.cells = np.zeros((self.input_shape, 3))
		self.cells[:, 0] = powers[2:self.input_shape+2, 0] # TT
		self.cells[:, 1] = powers[2:self.input_shape+2, 3] # TE
		self.cells[:, 2] = powers[2:self.input_shape+2, 1] # EE
		
		self.ells = np.arange(2, self.input_shape+2)
	
	def load_leakage(self, leak_filename, data_dir = ''):
		_, self.l98, self.l150 = np.loadtxt(data_dir + leak_filename, unpack = True)
		self.l98 = self.l98[:self.nbinte]
		self.l150 = self.l150[:self.nbinte]
		
		self.a1 = 1.0
		self.a2 = 1.0
	
	def cull_covmat(self):
		# We have now packed the covariance matrix and the window function matrix.
		# We want to ignore the first B data points, we do so by culling the covmat for each observation.
		for i in range(self.b0):
			for j in range(self.nspectt):
				# cull lmin in TT
				self.covmat[i+j*self.nbintt,:self.nbin] = 0.0
				self.covmat[:self.nbin,i+j*self.nbintt] = 0.0
				self.covmat[i+j*self.nbintt,i+j*self.nbintt] = 1e10
		
		for i in range(self.nspectt * self.nbintt, self.nspectt * self.nbintt + self.b1):
			for j in range(self.nspecte):
				# cull lmin in TE
				self.covmat[i+j*self.nbinte,:self.nbin] = 0.0
				self.covmat[:self.nbin,i+j*self.nbinte] = 0.0
				self.covmat[i+j*self.nbinte,i+j*self.nbinte] = 1e10
		
		for i in range(self.nspectt * self.nbintt + self.nspecte * self.nbinte, self.nspectt * self.nbintt + self.nspecte * self.nbinte + self.b2):
			for j in range(self.nspecee):
				# cull lmin in EE
				self.covmat[i+j*self.nbinee,:self.nbin] = 0.0
				self.covmat[:self.nbin,i+j*self.nbinee] = 0.0
				self.covmat[i+j*self.nbinee,i+j*self.nbinee] = 1e10
	
	def loglike(self, fg_tt = None, fg_te = None, fg_ee = None, yp1 = 1.0, yp2 = 1.0):
		if fg_tt is None and self.enable_tt:
			raise ValueError('TT foreground is expected but not given.')
		if fg_te is None and self.enable_te:
			raise ValueError('TE foreground is expected but not given.')
		if fg_ee is None and self.enable_ee:
			raise ValueError('EE foreground is expected but not given.')
		
		# Total C-ells.
		cltt = np.zeros((self.shape))
		clte = np.zeros((self.shape))
		clee = np.zeros((self.shape))
		
		cltt[:self.input_shape] = self.cells[:self.input_shape, 0]
		clte[:self.input_shape] = self.cells[:self.input_shape, 1]
		clee[:self.input_shape] = self.cells[:self.input_shape, 2]
		
		# CMB+Fg theory
		x_theory = np.zeros((self.nspec, self.shape))
		
		x_theory[0                         : self.nspectt                          ,:self.shape] = np.tile(cltt, (self.nspectt, 1)) + fg_tt
		x_theory[self.nspectt              : self.nspectt+self.nspecte             ,:self.shape] = np.tile(clte, (self.nspecte, 1)) + fg_te
		x_theory[self.nspectt+self.nspecte : self.nspectt+self.nspecte+self.nspecee,:self.shape] = np.tile(clee, (self.nspecee, 1)) + fg_ee
		
		ll = np.arange(2, self.shape+2)
		for i in range(x_theory.shape[0]):
			x_theory[i,:] = 2.0 * np.pi * x_theory[i,:] / (ll * (ll + 1.0))
		
		x_model = np.zeros((self.nbin))
		# Bin the model using the window functions
		
		# TT modes
		for j in range(self.nspectt):
			x_model[j * self.nbintt : (j+1) * self.nbintt] = self.win_func[j * self.nbintt : (j+1) * self.nbintt, :] @ x_theory[j,:] # TT
		
		# TE modes
		for j in range(self.nspecte):
			i0 = self.nspectt * self.nbintt
			j0 = self.nspectt
			x_model[i0 + j * self.nbinte : i0 + (j+1) * self.nbinte] = self.win_func[i0 + j * self.nbinte : i0 + (j+1) * self.nbinte, :] @ x_theory[j0+j,:] # TE
		
		# EE modes
		for j in range(self.nspecee):
			i0 = self.nspectt * self.nbintt + self.nspecte * self.nbinte
			j0 = self.nspectt + self.nspecte
			x_model[i0 + j * self.nbinee : i0 + (j+1) * self.nbinee] = self.win_func[i0 + j * self.nbinee : i0 + (j+1) * self.nbinee, :] @ x_theory[j0+j,:] # EE
		
		# Leakage
		if not self.l98 is None and not self.l150 is None:
			i0 = 3 * self.nbintt
			x_model[i0                   : i0 +     self.nbinte] += x_model[                :     self.nbintt] * self.a1 *  self.l98
			x_model[i0 +     self.nbinte : i0 + 2 * self.nbinte] += x_model[    self.nbintt : 2 * self.nbintt] * self.a2 * self.l150
			x_model[i0 + 2 * self.nbinte : i0 + 3 * self.nbinte] += x_model[    self.nbintt : 2 * self.nbintt] * self.a1 *  self.l98
			x_model[i0 + 3 * self.nbinte : i0 + 4 * self.nbinte] += x_model[2 * self.nbintt : 3 * self.nbintt] * self.a2 * self.l150
			
			i0 = 3 * self.nbintt + 4 * self.nbinte
			x_model[i0                   : i0 +     self.nbinee] += 2 * x_model[3 * self.nbintt                   : 3 * self.nbintt +     self.nbinte] * self.a1 * self.l98 + x_model[:self.nbintt] * np.power(self.a1 * self.l98, 2.0)
			x_model[i0 +     self.nbinee : i0 + 2 * self.nbinee] +=     x_model[3 * self.nbintt +     self.nbinte : 3 * self.nbintt + 2 * self.nbinte] * self.a1 * self.l98 + x_model[3 * self.nbintt + 2 * self.nbinte : 3 * self.nbintt + 3 * self.nbinte] * self.a2 * self.l150 + x_model[self.nbintt : 2 * self.nbintt] * self.a1 * self.l98 * self.a2 * self.l150
			x_model[i0 + 2 * self.nbinee : i0 + 3 * self.nbinee] += 2 * x_model[3 * self.nbintt + 3 * self.nbinte : 3 * self.nbintt + 4 * self.nbinte] * self.a2 * self.l150 + x_model[2 * self.nbintt : 3 * self.nbintt] * np.power(self.a2 * self.l150, 2.0)
		
		# Calibration
		x_model[                :     self.nbintt] = x_model[                :     self.nbintt] * self.ct1 * self.ct1
		x_model[    self.nbintt : 2 * self.nbintt] = x_model[    self.nbintt : 2 * self.nbintt] * self.ct1 * self.ct2
		x_model[2 * self.nbintt : 3 * self.nbintt] = x_model[2 * self.nbintt : 3 * self.nbintt] * self.ct2 * self.ct2
		
		i0 = self.nspectt * self.nbintt
		x_model[i0                   : i0 +     self.nbinte] = x_model[i0                   : i0 +     self.nbinte] * self.ct1 * self.ct1 * yp1
		x_model[i0 +     self.nbinte : i0 + 2 * self.nbinte] = x_model[i0 +     self.nbinte : i0 + 2 * self.nbinte] * self.ct1 * self.ct2 * yp2
		x_model[i0 + 2 * self.nbinte : i0 + 3 * self.nbinte] = x_model[i0 + 2 * self.nbinte : i0 + 3 * self.nbinte] * self.ct2 * self.ct1 * yp1
		x_model[i0 + 3 * self.nbinte : i0 + 4 * self.nbinte] = x_model[i0 + 3 * self.nbinte : i0 + 4 * self.nbinte] * self.ct2 * self.ct2 * yp2
		
		i0 = self.nspectt * self.nbintt + self.nspecte * self.nbinte
		x_model[i0                   : i0 +     self.nbinee] = x_model[i0                   : i0 +     self.nbinee] * self.ct1 * self.ct1 * yp1 * yp1
		x_model[i0 +     self.nbinee : i0 + 2 * self.nbinee] = x_model[i0 +     self.nbinee : i0 + 2 * self.nbinee] * self.ct1 * self.ct2 * yp1 * yp2
		x_model[i0 + 2 * self.nbinee : i0 + 3 * self.nbinee] = x_model[i0 + 2 * self.nbinee : i0 + 3 * self.nbinee] * self.ct2 * self.ct2 * yp2 * yp2
		
		subcov = self.covmat
		bin_no = self.nbin
		diff_vec = self.b_dat - x_model
		
		if self.enable_tt and not self.enable_te and not self.enable_ee:
			bin_no = self.nbintt * self.nspectt
			diff_vec = diff_vec[:bin_no]
			subcov = self.covmat[:bin_no,:bin_no]
			print('Using only TT.')
		elif not self.enable_tt and self.enable_te and not self.enable_ee:
			n0 = self.nbintt * self.nspectt
			bin_no = self.nbinte * self.nspecte
			diff_vec = diff_vec[n0:n0 + bin_no]
			subcov = self.covmat[n0:n0 + bin_no, n0:n0 + bin_no]
			print('Using only TE.')
		elif not self.enable_tt and not self.enable_te and self.enable_ee:
			n0 = self.nbintt * self.nspectt + self.nbinte * self.nspecte
			bin_no = self.nbinee * self.nspecee
			diff_vec = diff_vec[n0:n0 + bin_no]
			subcov = self.covmat[n0:n0 + bin_no, n0:n0 + bin_no]
			print('Using only EE.')
		elif self.enable_tt and self.enable_te and self.enable_ee:
			print('Using TT+TE+EE.')
		else:
			raise Exception('Improper combination of TT/TE/EE spectra selected.')
		
		fisher = linalg.cho_solve(linalg.cho_factor(subcov), b = np.identity(bin_no))
		
		tmp = fisher @ diff_vec
		return -np.dot(tmp, diff_vec) / 2.0
	
	def disable_leakage(self):
		self.a1 = 0.0
		self.a2 = 0.0
	
	def disable_calibration(self):
		self.ct1 = 0.0
		self.ct2 = 0.0
	
	@property
	def use_tt(self):
		return self.enable_tt
	
	@use_tt.setter
	def use_tt(self, val):
		self.enable_tt = val
	
	@property
	def use_te(self):
		return self.enable_te
	
	@use_te.setter
	def use_te(self, val):
		self.enable_te = val
	
	@property
	def use_ee(self):
		return self.enable_ee
	
	@use_ee.setter
	def use_ee(self, val):
		self.enable_ee = val
	
	@property
	def frequencies(self):
		return self.freqs
	
	@property
	def nbin(self):
		# total number of bins
		return self.nbintt * self.nspectt + self.nbinte * self.nspecte + self.nbinee * self.nspecee
	
	@property
	def nspec(self):
		# total number of spectra
		return self.nspectt + self.nspecte + self.nspecee
	
	@property
	def shape(self):
		return self.lmax_win-1
	
	@property
	def input_shape(self):
		return self.tt_lmax-1