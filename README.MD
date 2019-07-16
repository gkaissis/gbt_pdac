# A machine learning algorithm predicts molecular subtypes in pancreatic ductal adenocarcinoma with differential response to gemcitabine-based versus FOLFIRINOX chemotherapy

### Georgios Kaissis, Sebastian Ziegelmayer, Fabian Lohöfer, Katja Steiger, Hana Algül, Alexander Muckenhuber, Hsi-Yu Yen, Ernst Rummeny, Helmut Friess, Roland Schmid, Wilko Weichert, Jens T. Siveke and Rickmer Braren

This repository contains the data and code necessary to reproduce all results of the manuscript. Data is provided in the .csv format and as .dat files which can be loaded with the Python standard library tool ```joblib```.

This repository is indexed on Zenodo 

[![DOI](https://zenodo.org/badge/188212363.svg)](https://zenodo.org/badge/latestdoi/188212363)

---
### Installation instructions

1. Create a new python environment
2. Install ```requirements.txt``` with pip install -r requirements.txt
3. The ```analysis.py``` file contains the code for the analysis. It needs to be modified to select whether .dat or .csv is loaded by uncommenting the appropriate lines. Please observe that .dat files can contain executable code which is run when loaded and should always be checked for safety before loading. We assume no responsibility for damage due to improper use.
4. The file ```clinical_data.csv``` contains the clinical data. Indexes match the radiomics indexes.

---
### File Index

```Features.csv/features.dat```: Reproducible and stable features (N=50) derived from the ADC maps alongside feature names and patient indices

```Labels.csv/labels.dat```: Patient labels (0= KRT81 negative, 1= KRT 81 positive)

```clinical_data.csv```: Clinical patient data

```requirements.txt```: Required python libraries with pinned version numbers

```analysis.py```: code used for generating the results

---
Please note that we are unable to offer technical support. However, please feel free to offer scientific input or additional ideas via issue or pull request.
