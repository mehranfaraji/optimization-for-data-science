# Training Soft-Margin SVMs using Frank-Wolfe, Away-Step Frank-Wolfe, and Pairwise Frank-Wolfe

## Project Overview

This repository contains the implementation and experimentation for training Soft-Margin SVMs using three different optimization algorithms: Frank-Wolfe, Away-Step Frank-Wolfe, and Pairwise Frank-Wolfe. The project explores their performance on various datasets and provides insights into their effectiveness for SVM training.

## Authors

- **Mehran Farajinegarestan**
  - *Department of Mathematics, University of Padova*
  - *Email: mehran.farajinegarestan@studenti.unipd.it*

- **Nazanin Karimi**
  - *Department of Mathematics, University of Padova*
  - *Email: Nazanin.Karimi@studenti.unipd.it*

## Files in the Repository

- `datasets.py`: Python module for handling datasets.
- `experiments.ipynb`: Jupyter notebook containing experiments and demonstration of training models.
- `FW.py`: Implementation of the Frank-Wolfe optimization algorithm and its variants.
- `Optimization_report.pdf`: Report of the project.
- `plots.py`: Python module used for generating plots.
- `images/`: Directory containing images produced in the experiments.ipynb.
- `datasets/`: Directory containing datasets used for the project.

## How to Run Experiments

1. Open the `experiments.ipynb` notebook in a Jupyter environment.
2. Run each cell in the notebook to execute experiments and analyze results.

## Additional Information

Note that training models on *a4a* and *SVM guide* datasets is time consuming. You can either reduce the number of iterations or use *Liver Disorder* or *Synthetic* dataset. 
