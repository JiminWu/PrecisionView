# PrecisionView
This is an open source repository of the PrecisionView ([link](to be added)).

"Deep learning dual-modality endomicroscope with large field-of-view and depth-of-field for real-time in vivo mapping of epithelial morphology", Huayu Hou<sup>&</sup>, Jimin Wu<sup>&</sup>, Jinyun Liu, Vivek Boominathan, Argaja Shende, Karthik Goli, Jennifer Carns, Richard A. Schwarz, Ann M. Gillenwater, Preetha Ramalingam, Mila P. Salcedo, Kathleen M. Schmeler, Tomasz S. Tkaczyk, Jacob T. Robinson, Ashok Veeraraghavan* and Rebecca R. Richards-Kortum*

<sub><sup>&</sup> Denotes equal contribution. | * Corresponding authors </sub>

## Hardware design
The directory 'PrecisionView CAD Files' contains the CAD files of the system, including the system housing and the distal cover. The file 'Precision Assembly' provides an assembly of the system with the illumination and the distal cover (Figure 2a). 

The directory 'Phase Mask Fabrication' contains the height map generated from the end-to-end training.

## System requirements
Python 3.8+
Environment file available as 'environment.yml'

## End-to-end training 

* Step1:

  Two supporting fata files are necessary for fast reconstruction:
  1. A single point spread function (PSF) file, saved as a '.mat' file.
  2. Raw capture from Bio-FlatScopeNHP, saved as a '.tiff' image file.

  We have provided examples for fast reconstruction test in the directory 'Bio-FlatScopeNHP Reconstruction/Examples'. 

* Step2

  Three supporting data files are necessary for spatially-varying reconstruction:
  1. A '.mat' file including all the registered PSFs.
  2. A '.mat' file including all the spatially-varying weights.
  3. Raw capture from Bio-FlatScopeNHP, saved as a '.tiff' image file.
  4. 

## Finetuning using captured PSFs

The directory 'In Vivo Data Processing Code' contains the customized MATLAB code for processing the position tuning data and orientation columns maps.
Data processing pipelines are the same for ground truth data and Bio-FlatScopeNHP data. Supporting functions can be found in the directory 'In Vivo Data Processing Code/Tools'. References of the data processing code:
* Chen, Y., Geisler, W. S. & Seidemann, E. Optimal decoding of correlated neural population responses in the primate visual cortex. Nat. Neurosci. 9, 1412–1420 (2006).
* Palmer, C. R., Chen, Y. & Seidemann, E. Uniform spatial spread of population activity in primate parafoveal V1. J. Neurophysiol. 107, 1857–1867 (2012).
* Seidemann, E. et al. Calcium imaging with genetically encoded indicators in behaving primates. eLife 5, e16178 (2016).
* Benvenuti, G. et al. Scale-Invariant Visual Capabilities Explained by Topographic Representations of Luminance and Texture in Primate V1. Neuron 100, 1504-1512.e4 (2018).

## Contact Us
In case of any queries regarding the code, please reach out to [Jimin](mailto:jimin.wu@rice.edu) or [Huayu](mailto:hhou@rice.edu).
Other raw and analysed data are available for research purpose from corresponding author upon reasonable request.
