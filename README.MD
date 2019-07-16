# A machine learning algorithm predicts molecular subtypes in pancreatic ductal adenocarcinoma with differential response to gemcitabine-based versus FOLFIRINOX chemotherapy

### Georgios Kaissis, Sebastian Ziegelmayer, Fabian Lohöfer, Katja Steiger, Hana Algül, Alexander Muckenhuber, Hsi-Yu Yen, Ernst Rummeny, Helmut Friess, Roland Schmid, Wilko Weichert, Jens T. Siveke and Rickmer Braren

This repository contains the data and code for this publication. Data is provided in the .csv format and as .dat files which can be loaded with the Python standard library tool ```joblib```.

---
### Installation instructions

1. Create a new python environment
2. Install ```requirements.txt``` with pip install -r requirements.txt
3. The ```analysis.py``` file contains the code for the analysis. It needs to be modified to select whether .dat or .csv is loaded by uncommenting the appropriate lines. Please observe that .dat files can contain executable code which is run when loaded and should always be checked for safety before loading. We assume no responsibility for damage due to improper use.
4. The file ```clinical_data.csv``` contains the clinical data. Indexes match the radiomics indexes.

---
Please note that we are unable to offer technical support. However, please feel free to offer scientific input or additional ideas via issue or pull request.