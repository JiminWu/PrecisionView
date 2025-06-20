# PrecisionView
This is an open source repository of the PrecisionView ([link](to be added)).

"Deep learning dual-modality endomicroscope with large field-of-view and depth-of-field for real-time in vivo mapping of epithelial morphology", Huayu Hou<sup>&</sup>, Jimin Wu<sup>&</sup>, Jinyun Liu, Vivek Boominathan, Argaja Shende, Karthik Goli, Jennifer Carns, Richard A. Schwarz, Ann M. Gillenwater, Preetha Ramalingam, Mila P. Salcedo, Kathleen M. Schmeler, Tomasz S. Tkaczyk, Jacob T. Robinson, Ashok Veeraraghavan* and Rebecca R. Richards-Kortum*

<sub><sup>&</sup> Denotes equal contribution. | * Corresponding authors </sub>

## Hardware design
The directory 'PrecisionView CAD Files' contains the CAD files of the system, including the system housing and the distal cover. The file 'Precision Assembly' provides an assembly of the system with the illumination and the distal cover (Figure 2a). 

The directory 'Phase Mask Fabrication' contains the height map generated from the end-to-end training.

## System requirements
Python 3.8+

Required packages and versions can be found in environment.yml. It can also be used to create a conda environment.

## End-to-end training and finetune
The directory 'E2E_Optimization' contains the code for the end-to-end training. Two supporting data files ('.mat') are necessary phase mask profile initialization.

* Step1: does not update the optical layer and only trains the digital layer.
* Step2: jointly update the optical layer and the digital layer.

The directory 'Finetune' contains the code for funetuning using real captured PSFs. 

## Contact Us
In case of any queries regarding the code, please reach out to [Jimin](mailto:jimin.wu@rice.edu) or [Huayu](mailto:hhou@rice.edu).
Other raw and analysed data are available for research purpose from corresponding author upon reasonable request.
