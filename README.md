# loglike
Calculate the log-likelihood for an ACT foreground model.

This is a quick (and somewhat dirty) recode of most of the chi^2 functionality of the original **DR4 Baseline Multi-Frequency Likelihood** from ACTPol, available [here](https://lambda.gsfc.nasa.gov/product/act/act_dr4_likelihood_multi_info.cfm).

## Requirements

You need `numpy`, `scipy` and python3 installed to run this. For additional functionality, you also need `camb` and/or `sacc` installed (see links below).

## Usage

`import loglike` in your python script. You must create a `Likelihood` instance for access to the functionality, e.g. `like = loglike.Likelihood()`. You can set the number of bins for the TT/TE/EE spectra by directly accessing the `like.nbintt` parameters, and the number of spectra used by directly accessing the `like.nspectt` parameters (similar for te and ee). The classes automatically assign the default values used by the FORTRAN code.

You can load in a plaintext dataset by calling `like.load_plaintext()`. It has a number of parameters, referring to the three files (model spectra, covariance matrix, window functions) and an optional path to where these files are loaded in. You can load in a dataset from a [SACC file](https://sacc.readthedocs.io/en/latest/sacc.html) by calling `like.load_sacc()`. Note that loading from a plaintext assumes _you_ set all parameters such as number of spectra and bins beforehand, whereas loading from a SACC file will make the `Likelihood` class overwrite values with what is found in the SACC file. You can load in a (plaintext) leakage dataset by calling `like.load_leakage()`.

You can load in a LCDM model by calling either `like.load_cells()` (to load it from a plaintext file) or by calling `like.load_cells_camb()` to have a background cosmology generated via [camb](https://camb.readthedocs.io/en/latest/).

When all is prepared and done, you can calculate the log-likelihood for a foreground model by calling `like.loglike()`. You have to provide TT/TE/EE foreground models (you can disable individual models by setting `like.use_ee = False` for example, but note not every combination of TT/TE/EE is allowed).

## Code maintenance

This code is presented as-is with no warranty for its functionality. Please refer to the original FORTRAN code if you wonder about the underlying functionality.

This code came to be by stitching together a great variety of different codesets, and thus may not be entirely stable, optimal or clean. The original author may not update this code in the (nearby) future, and they also cannot guarantee bug-free functionality.

If you find any bugs and you _desperately_ want to push a commit to patch it, feel free to contact the original author.

## License

This code was made by Hidde Jense and is available for usage and modification free of charge. Please contact Hidde Jense if you wish to incorporate this code into your own project(s). You are free to use and modify this code for personal or academical use, so long as you refer to the original author when you do so.

## References

Based on original code presented in Choi et al. 2020 and Aiola et al. 2020. Original code available in FORTRAN [here](https://lambda.gsfc.nasa.gov/product/act/act_dr4_likelihood_multi_info.cfm). Please reference them if you use their code at a later stage.

SACC is created by Joe Zuntz and is available [here](https://sacc.readthedocs.io/en/latest/sacc.html).

CAMB is created by Antony Lewis et al., and is available [here](https://camb.readthedocs.io/en/latest/).
