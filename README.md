# CS5850-Neural3D
Project for CS 5850, Neural Network Approach for 3D Genome Reconstruction

Justin Leo

Steps to Run Code:
Need to Provide 3 arguments: matrix for training, pdb for training, matrix for testing
Put matrix files in: NeuralRun_Data/Matrix_Data
Put PDB file in: NeuralRun_Data/Train_Structures

Format to Run:
	python Main.py -training matrix- -training pdb- -testing matrix-
Generated structure will be for testing matrix data

Example Run For Tested Dataset, this data is on Git:
python Main.py regular90.txt best_structure_regular90_IF.pdb regular70.txt

All Generated Outputs will be in:
	Generated_Outputs

Note, these tests are performed with a GPU in the system. Without a GPU, the run time will be very long.

Code Dependency Packages:
	numpy, keras, random, math, scipy, sklearn

Get Dependencies by:
	pip install -package-

