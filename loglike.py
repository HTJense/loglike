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
		
		# Total number of bins per cross spectrum
		self.nbintt = []
		self.nbinte = []
		self.nbinee = []
		
		# Which frequencies are crossed per spectrum
		self.crosstt = []
		self.crosste = []
		self.crossee = []
		
		self.freqs = [ 95, 150 ]
		
		# maximum ell for windows and for cross spectra
		self.lmax_win = 7925
		self.tt_lmax = 6000
		
		# amount of low-ell bins to ignore per cross spectrum
		self.b0 = 5
		self.b1 = 0
		self.b2 = 0
		
		# calibration parameters
		self.ct = [ 1.0 for _ in self.freqs ]
		self.yp = [ 1.0 for _ in self.freqs ]
		
		# leakage terms
		self.a1 = 0.0
		self.a2 = 0.0
		self.l98 = None
		self.l150 = None
		
		# Whether or not to use TT, EE and/or TE (or all)
		self.enable_tt = True
		self.enable_te = True
		self.enable_ee = True
	
	def clear(self):
		self.win_func = None
		self.covmat = None
		self.cells = None
		self.b_dat = None
		self.win_ells = None
		self.b_ell = None
		self.ells = None
		
		self.nbintt = []
		self.nbinte = []
		self.nbinee = []
		
		self.crosstt = []
		self.crosste = []
		self.crossee = []
		
		self.freqs = [ ]
		
		self.lmax_win = 0
		self.tt_lmax = 0
		
		self.b0 = 0
		self.b1 = 0
		self.b2 = 0
		
		self.ct = [ ]
		self.yp = [ ]
		
		self.a1 = 0.0
		self.a2 = 0.0
		self.l98 = None
		self.l150 = None
		
		self.enable_tt = True
		self.enable_te = True
		self.enable_ee = True
	
	def load_plaintext(self, spec_filename, cov_filename, bbl_filename, bins, cross, lmax_win = None, data_dir = ''):
		if self.nbin == 0:
			# We don't yet have a model loaded in, so we can just take all given data.
			self.nbintt = bins[0]
			self.nbinte = bins[1]
			self.nbinee = bins[2]
			
			self.crosstt = cross[0]
			self.crosste = cross[1]
			self.crossee = cross[2]
			
			if not lmax_win is None:
				self.lmax_win = lmax_win
			
			self.win_func = np.loadtxt(data_dir + bbl_filename)[:self.nbin,:self.lmax_win]
			self.b_dat = np.loadtxt(data_dir + spec_filename)[:self.nbin]
			
			self.covmat = np.loadtxt(data_dir + cov_filename, dtype = float)[:self.nbin,:self.nbin]
		else:
			# We need to insert this data into the existing data
			if lmax_win is None:
				lmax_win = self.lmax_win
			 
			newbin = [ sum(bins[i]) for i in range(len(bins)) ]
			new_win = np.loadtxt(data_dir + bbl_filename)[:sum(newbin), :lmax_win]
			
			if lmax_win < self.lmax_win:
				# The existing data extends to higher ell than the provided data,
				# thus we pad the provided data with zeros.
				result = np.zeros((sum(newbin), self.lmax_win))
				result[:,:lmax_win] = new_win
				new_win = result
			elif lmax_win > self.lmax_win:
				# The provided data extends to higher ell than the existing data,
				# thus we pad the existing data with zeros.
				result = np.zeros((self.nbin, lmax_win))
				result[:,:self.lmax_win] = self.win_func
				self.win_func = result
				self.lmax_win = lmax_win
			
			# We now know for sure that both window arrays have the same number of ell bins (i.e. their ndim[1] is equal).
			# Now we can zip them together.
			
			# First we need to split either window function into separate TT/TE/EE components.
			old_tt, old_te, old_ee = self.win_func[:sum(self.nbintt),:], self.win_func[sum(self.nbintt):sum(self.nbintt) + sum(self.nbinte),:], self.win_func[sum(self.nbintt)+sum(self.nbinte):sum(self.nbintt)+sum(self.nbinte)+sum(self.nbinee),:]
			new_tt, new_te, new_ee = new_win[:newbin[0],:], new_win[newbin[0]:newbin[0]+newbin[1],:], new_win[newbin[0]+newbin[1]:newbin[0]+newbin[1]+newbin[2],:]
			# And we then stack them in the correct order and merge
			self.win_func = np.concatenate((old_tt, new_tt, old_te, new_te, old_ee, new_ee), axis = 0)
			
			# Now we do the same for the covariance matrix.
			cov2 = np.loadtxt(data_dir + cov_filename, dtype = float)[:sum(newbin),:sum(newbin)]
			oldbin = [ sum(self.nbintt), sum(self.nbinte), sum(self.nbinee) ]
			
			ncov = np.zeros((sum(oldbin)+sum(newbin), sum(oldbin)+sum(newbin)))
			
			# This is some "clever" interlacing of the two covariance matrices.
			for i in range(3):
				for j in range(3):
					ncov[ sum(oldbin[:i  ]) + sum(newbin[:i]) : sum(oldbin[:i+1]) + sum(newbin[:i  ]), sum(oldbin[:j  ]) + sum(newbin[:j]) : sum(oldbin[:j+1]) + sum(newbin[:j  ]) ] = self.covmat[ sum(oldbin[:i]) : sum(oldbin[:i+1]), sum(oldbin[:j]) : sum(oldbin[:j+1]) ]
					ncov[ sum(oldbin[:i+1]) + sum(newbin[:i]) : sum(oldbin[:i+1]) + sum(newbin[:i+1]), sum(oldbin[:j+1]) + sum(newbin[:j]) : sum(oldbin[:j+1]) + sum(newbin[:j+1]) ] =        cov2[ sum(newbin[:i]) : sum(newbin[:i+1]), sum(newbin[:j]) : sum(newbin[:j+1]) ]
			
			self.covmat = ncov
			
			# And finally for the data vector.
			n_dat = np.loadtxt(data_dir + spec_filename)[:sum(newbin)]
			old_tt, old_te, old_ee = self.b_dat[:sum(self.nbintt)], self.b_dat[sum(self.nbintt):sum(self.nbintt)+sum(self.nbinte)], self.b_dat[sum(self.nbintt)+sum(self.nbinte):sum(self.nbintt)+sum(self.nbinte)+sum(self.nbinee)]
			new_tt, new_te, new_ee = n_dat[:newbin[0]], n_dat[newbin[0]:newbin[0]+newbin[1]], n_dat[newbin[0]+newbin[1]:newbin[0]+newbin[1]+newbin[2]]
			
			self.b_dat = np.concatenate((old_tt, new_tt, old_te, new_te, old_ee, new_ee), axis = 0)
			
			# Now that all is well, we can add the new bins to the old bins.
			# The cross indices are first shifted over.
			self.crosstt += cross[0]
			self.crosste += cross[1]
			self.crossee += cross[2]
			
			self.nbintt += bins[0]
			self.nbinte += bins[1]
			self.nbinee += bins[2]
	
	def bins_from_sacc(self, saccfile, xp_name = 'LAT'):
		# Check if the numbers add up, we expect N(N+1)/2 for TT/EE and N^2 for TE.
		n = int(np.sqrt(len(saccfile.get_tracer_combinations('cl_0e'))))
		ntt = n * (n + 1) // 2
		nte = n * n
		
		# List all used frequencies.
		# casting a list to a set back to a list to filter out only unique values.
		matcher = re.compile('{}_([0-9]+)_([02s]+)'.format(xp_name))
		
		self.freqs = sorted(list(set([ int(matcher.search(x).groups()[0]) for x,_ in saccfile.get_tracer_combinations('cl_00') ])))
		self.ct = [ 1.0 for _ in self.freqs ]
		self.yp = [ 1.0 for _ in self.freqs ]
		
		self.nbintt = []
		self.nbinte = []
		self.nbinee = []
		
		self.crosstt = []
		self.crosste = []
		self.crossee = []
		
		# We check how large our window function + covmat should be
		for i in range(len(saccfile.get_tracer_combinations('cl_00'))):
			tt1, tt2 = saccfile.get_tracer_combinations('cl_00')[i]
			_, tmp_cells = saccfile.get_ell_cl('cl_00', tt1, tt2, return_cov = False)
			self.nbintt.append(tmp_cells.shape[0])
			
			i1 = int(matcher.search(tt1).groups()[0])
			i2 = int(matcher.search(tt2).groups()[0])
			self.crosstt.append((self.freqs.index(i1), self.freqs.index(i2)))
		
		for i in range(len(saccfile.get_tracer_combinations('cl_ee'))):
			ee1, ee2 = saccfile.get_tracer_combinations('cl_ee')[i]
			_, tmp_cells = saccfile.get_ell_cl('cl_ee', ee1, ee2, return_cov = False)
			self.nbinee.append(tmp_cells.shape[0])
			
			i1 = int(matcher.search(ee1).groups()[0])
			i2 = int(matcher.search(ee2).groups()[0])
			self.crossee.append((self.freqs.index(i1), self.freqs.index(i2)))
		
		for j in range(len(saccfile.get_tracer_combinations('cl_0e'))):
			te1, te2 = saccfile.get_tracer_combinations('cl_0e')[j]
			_, tmp_cells = saccfile.get_ell_cl('cl_0e', te1, te2, return_cov = False)
			self.nbinte.append(tmp_cells.shape[0])
			
			i1 = int(matcher.search(te1).groups()[0])
			i2 = int(matcher.search(te2).groups()[0])
			
			# Dirty: TE/ET ordering matters, so we need to check in what order the sacc-file gives the tracers to us.
			# The code *asserts* that every spectrum is in TE-ordering, so if we find an ET-ordered band, we cheat and swap the frequencies here.
			if matcher.search(te1).groups()[1] == 's2':
				i1, i2 = i2, i1
			
			self.crosste.append((self.freqs.index(i1), self.freqs.index(i2)))
		
		if self.nspectt != ntt or self.nspecte != nte or self.nspecee != ntt:
			raise ValueError('Incorrect number of spectra found: expected {}+{}+{} but found {}+{}+{} (TT+TE+EE).'.format(ntt, nte, ntt, self.nspectt, self.nspecte, self.nspecee))
	
	def covmat_from_sacc(self, saccfile, xp_name = 'LAT'):
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
		
		# We keep track of which index goes where to properly stack the covariance matrix.
		w_ind = np.zeros((self.nbin,), dtype = int)
		
		# We now add in all the 
		for j, (x1, x2) in enumerate(self.crosstt):
			f1 = self.freqs[x1]
			f2 = self.freqs[x2]
			eltt, cltt, covtt, ind_tt = saccfile.get_ell_cl('cl_00', '{}_{}_s0'.format(xp_name, f1), '{}_{}_s0'.format(xp_name, f2), return_cov = True, return_ind = True)
			win_tt = saccfile.get_bandpower_windows(ind_tt)
			
			n0 = sum(self.nbintt[0:j])
			self.b_dat[n0:n0+self.nbintt[j]] = cltt
			self.b_ell[n0:n0+self.nbintt[j]] = eltt
			w_ind[n0:n0+self.nbintt[j]] = ind_tt
			self.win_func[n0:n0+self.nbintt[j],:] = win_tt.weight[:,:].T
		
		for j, (x1, x2) in enumerate(self.crossee):
			f1 = self.freqs[x1]
			f2 = self.freqs[x2]
			
			elee, clee, covee, ind_ee = saccfile.get_ell_cl('cl_ee', '{}_{}_s2'.format(xp_name, f1), '{}_{}_s2'.format(xp_name, f2), return_cov = True, return_ind = True)
			win_ee = saccfile.get_bandpower_windows(ind_ee)
			
			n0 = sum(self.nbintt) + sum(self.nbinte) + sum(self.nbinee[0:j])
			self.b_dat[n0:n0+self.nbinee[j]] = clee
			self.b_ell[n0:n0+self.nbinee[j]] = elee
			w_ind[n0:n0+self.nbinee[j]] = ind_ee
			self.win_func[n0:n0+self.nbinee[j],:] = win_ee.weight[:,:].T
		
		for i, (x1, x2) in enumerate(self.crosste):
			f1 = self.freqs[x1]
			f2 = self.freqs[x2]
			n0 = sum(self.nbintt) + sum(self.nbinte[0:i])
			
			elte, clte, covte, ind_te = saccfile.get_ell_cl('cl_0e', '{}_{}_s0'.format(xp_name, f1), '{}_{}_s2'.format(xp_name, f2), return_cov = True, return_ind = True)
			
			if covte.shape == (0,0):
				# Sometimes the tracers are in the other order (they always store them such that F1 < F2), so we gotta reverse the order
				# Remember that we force the sort the other way around (see above when we loaded in the spectra frequencies).
				elte, clte, covte, ind_te = saccfile.get_ell_cl('cl_0e', '{}_{}_s2'.format(xp_name, f2), '{}_{}_s0'.format(xp_name, f1), return_cov = True, return_ind = True)
			
			win_te = saccfile.get_bandpower_windows(ind_te)
			self.b_dat[n0:n0+self.nbinte[i]] = clte
			self.b_ell[n0:n0+self.nbinte[i]] = elte
			w_ind[n0:n0+self.nbinte[j]] = ind_te
			self.win_func[n0:n0+self.nbinte[i],:] = win_te.weight[:,:].T
		
		# We now make sure to properly index the saccfile's covariance matrix into the one we need.
		self.covmat[:,:] = saccfile.covariance.covmat[w_ind,:][:,w_ind]
	
	def load_sacc(self, sacc_filename, data_dir = '', xp_name = 'LAT'):
		try:
			import sacc
		except ImportError as e:
			print('Failed to load data from a SACC file: failed to import sacc.\n{}'.format(str(e)))
			return
		
		saccfile = sacc.Sacc.load_fits(data_dir + sacc_filename)
		
		self.bins_from_sacc(saccfile, xp_name = xp_name)
		self.covmat_from_sacc(saccfile, xp_name = xp_name)
		
		# TODO: Make some method/overview of what each index should represent (i.e. "index 0 should represent ell = 2 for the 95x95 TT spectrum, index 1 should be (3,95x95,TT), etc...)
		
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
		# Note: this assumes all TE bins are the same length (it is hardcoded at a later point).
		_, self.l98, self.l150 = np.loadtxt(data_dir + leak_filename, unpack = True)
		self.l98 = self.l98[ :self.nbinte[0] ]
		self.l150 = self.l150[ :self.nbinte[0] ]
		
		self.a1 = 1.0
		self.a2 = 1.0
	
	def cull_covmat(self):
		# We have now packed the covariance matrix and the window function matrix.
		# We want to ignore the first B data points, we do so by culling the covmat for each observation.
		for i in range(self.b0):
			for j in range(self.nspectt):
				# cull lmin in TT
				self.covmat[i+sum(self.nbintt[0:j]),:self.nbin] = 0.0
				self.covmat[:self.nbin,i+sum(self.nbintt[0:j])] = 0.0
				self.covmat[i+sum(self.nbintt[0:j]),i+sum(self.nbintt[0:j])] = 1e10
		
		for i in range(sum(self.nbintt), sum(self.nbintt) + self.b1):
			for j in range(self.nspecte):
				# cull lmin in TE
				self.covmat[i+sum(self.nbinte[0:j]),:self.nbin] = 0.0
				self.covmat[:self.nbin,i+sum(self.nbinte[0:j])] = 0.0
				self.covmat[i+sum(self.nbinte[0:j]),i+sum(self.nbinte[0:j])] = 1e10
		
		for i in range(sum(self.nbintt) + sum(self.nbinte), sum(self.nbintt) + sum(self.nbinte) + self.b2):
			for j in range(self.nspecee):
				# cull lmin in EE
				self.covmat[i+sum(self.nbinee[0:j]),:self.nbin] = 0.0
				self.covmat[:self.nbin,i+sum(self.nbinee[0:j])] = 0.0
				self.covmat[i+sum(self.nbinee[0:j]),i+sum(self.nbinee[0:j])] = 1e10
	
	def loglike(self, fg_tt = None, fg_te = None, fg_ee = None):
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
			x_model[sum(self.nbintt[0:j]) : sum(self.nbintt[0:j+1])] = self.win_func[sum(self.nbintt[0:j]) : sum(self.nbintt[0:j+1]), :] @ x_theory[j,:] # TT
		
		# TE modes
		for j in range(self.nspecte):
			i0 = sum(self.nbintt)
			j0 = self.nspectt
			x_model[i0 + sum(self.nbinte[0:j]) : i0 + sum(self.nbinte[0:j+1])] = self.win_func[i0 + sum(self.nbinte[0:j]) : i0 + sum(self.nbinte[0:j+1]), :] @ x_theory[j0+j,:] # TE
		
		# EE modes
		for j in range(self.nspecee):
			i0 = sum(self.nbintt) + sum(self.nbinte)
			j0 = self.nspectt + self.nspecte
			x_model[i0 + sum(self.nbinee[0:j]) : i0 + sum(self.nbinee[0:j+1])] = self.win_func[i0 + sum(self.nbinee[0:j]) : i0 + sum(self.nbinee[0:j+1]), :] @ x_theory[j0+j,:] # EE
		
		# Leakage
		if not self.l98 is None and not self.l150 is None:
			# TODO: Leakage for nfreq > 2.
			# Currently it is hardcoded and only allows for 2 freq leakage, but it should be changed to allow for n > 2.
			
			# Modify TE spectra by adding scaled TT components.
			i0 = sum(self.nbintt)
			x_model[i0 + sum(self.nbinte[0:0]) : i0 + sum(self.nbinte[0:1]) ] += x_model[ sum(self.nbintt[0:0]) : sum(self.nbintt[0:1]) ] * self.a1 * self.l98
			x_model[i0 + sum(self.nbinte[0:1]) : i0 + sum(self.nbinte[0:2]) ] += x_model[ sum(self.nbintt[0:1]) : sum(self.nbintt[0:2]) ] * self.a2 * self.l150
			x_model[i0 + sum(self.nbinte[0:2]) : i0 + sum(self.nbinte[0:3]) ] += x_model[ sum(self.nbintt[0:1]) : sum(self.nbintt[0:2]) ] * self.a1 * self.l98
			x_model[i0 + sum(self.nbinte[0:3]) : i0 + sum(self.nbinte[0:4]) ] += x_model[ sum(self.nbintt[0:2]) : sum(self.nbintt[0:3]) ] * self.a2 * self.l150
			
			# Modify EE spectra by adding scaled TT/TE components.
			j0 = sum(self.nbintt) + sum(self.nbinte)
			x_model[j0 + sum(self.nbinee[0:0]) : j0 + sum(self.nbinee[0:1]) ] += 2 * x_model[ i0 + sum(self.nbinte[0:0]) : i0 + sum(self.nbinte[0:1]) ] * self.a1 * self.l98
			x_model[j0 + sum(self.nbinee[0:0]) : j0 + sum(self.nbinee[0:1]) ] +=     x_model[      sum(self.nbintt[0:0]) :      sum(self.nbintt[0:1]) ] * self.a1 * self.l98 * self.a1 * self.l98
			
			x_model[j0 + sum(self.nbinee[0:1]) : j0 + sum(self.nbinee[0:2]) ] +=     x_model[ i0 + sum(self.nbinte[0:1]) : i0 + sum(self.nbinte[0:2]) ] * self.a1 * self.l98
			x_model[j0 + sum(self.nbinee[0:1]) : j0 + sum(self.nbinee[0:2]) ] +=     x_model[ i0 + sum(self.nbinte[0:2]) : i0 + sum(self.nbinte[0:3]) ] * self.a2 * self.l150
			x_model[j0 + sum(self.nbinee[0:1]) : j0 + sum(self.nbinee[0:2]) ] +=     x_model[      sum(self.nbintt[0:1]) :      sum(self.nbintt[0:2]) ] * self.a1 * self.l98 * self.a2 * self.l150
			
			x_model[j0 + sum(self.nbinee[0:2]) : j0 + sum(self.nbinee[0:3]) ] += 2 * x_model[ i0 + sum(self.nbinte[0:3]) : i0 + sum(self.nbinte[0:4]) ] * self.a2 * self.l150
			x_model[j0 + sum(self.nbinee[0:0]) : j0 + sum(self.nbinee[0:1]) ] +=     x_model[      sum(self.nbintt[0:2]) :      sum(self.nbintt[0:3]) ] * self.a2 * self.l150 * self.a2 * self.l150
		
		# Calibration
		for i in np.arange(len(self.nbintt)):
			# Mode T[i]xT[j] should be calibrated using CT[i] * CT[j]
			m1, m2 = self.crosstt[i]
			x_model[ sum(self.nbintt[0:i]) : sum(self.nbintt[0:i+1]) ] = x_model[ sum(self.nbintt[0:i]) : sum(self.nbintt[0:i+1]) ] * self.ct[m1] * self.ct[m2]
		
		for i in np.arange(len(self.nbinte)):
			# Mode T[i]xE[j] should be calibrated using CT[i] * (CT[j]*YP[j])
			m1, m2 = self.crosste[i]
			i0 = sum(self.nbintt)
			x_model[ i0 + sum(self.nbinte[0:i]) : i0 + sum(self.nbinte[0:i+1]) ] = x_model[ i0 + sum(self.nbinte[0:i]) : i0 + sum(self.nbinte[0:i+1]) ] * self.ct[m1] * (self.ct[m2] * self.yp[m2])
		
		for i in np.arange(len(self.nbinee)):
			# Mode E[i]xE[j] should be calibrated using (CT[i]*YP[i]) * (CT[j]*YP[j])
			m1, m2 = self.crossee[i]
			i0 = sum(self.nbintt) + sum(self.nbinte)
			x_model[ i0 + sum(self.nbinee[0:i]) : i0 + sum(self.nbinee[0:i+1]) ] = x_model[ i0 + sum(self.nbinee[0:i]) : i0 + sum(self.nbinee[0:i+1]) ] * (self.ct[m1] * self.yp[m1]) * (self.ct[m2] * self.yp[m2])
		
		subcov = self.covmat
		bin_no = self.nbin
		diff_vec = self.b_dat - x_model
		
		if self.enable_tt and not self.enable_te and not self.enable_ee:
			bin_no = sum(self.nbintt)
			diff_vec = diff_vec[:bin_no]
			subcov = self.covmat[:bin_no,:bin_no]
			print('Using only TT.')
		elif not self.enable_tt and self.enable_te and not self.enable_ee:
			n0 = sum(self.nbintt)
			bin_no = sum(self.nbinte)
			diff_vec = diff_vec[n0:n0 + bin_no]
			subcov = self.covmat[n0:n0 + bin_no, n0:n0 + bin_no]
			print('Using only TE.')
		elif not self.enable_tt and not self.enable_te and self.enable_ee:
			n0 = sum(self.nbintt) + sum(self.nbinte)
			bin_no = sum(self.nbinee)
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
	def nspectt(self):
		return len(self.nbintt)
	
	@property
	def nspecte(self):
		return len(self.nbinte)
	
	@property
	def nspecee(self):
		return len(self.nbinee)
	
	@property
	def nbin(self):
		# total number of bins
		return sum(self.nbintt) + sum(self.nbinte) + sum(self.nbinee)
	
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
