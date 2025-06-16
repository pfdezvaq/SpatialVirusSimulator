# Replication Dynamics of Herpesvirus-Derived Vectors
====================================================

This repository contains a spatial simulation model for analyzing the replication dynamics
of engineered herpesvirus vectors, including co-infection dynamics between a WT
and an amplicon. The model captures:

- Infection dynamics in a 2D culture plate
- Virion diffusion, degradation, and production
- Cell states: uninfected, WT-infected, amplicon-infected, coinfected
- Multi-generational simulations with evolutionary constraints
- Clustering analysis of infected cells

Main Files
----------
- `spatial_virus_simulator.py`: Core simulator class and clustering/evolution utilities.
- `main.py`: Example usage scripts for running single simulations, multi-generations, and clustering analysis.

Requirements
------------
- Python 3.7+
- NumPy, Matplotlib, SciPy, Seaborn, Pandas

To install dependencies:

    pip install numpy matplotlib scipy seaborn pandas

Citation and Attribution
------------------------
If you use this model or build upon it for academic, experimental, or publication purposes,
**you must cite the original author**:

**Fernández Vaquero, Pablo & Lim, Filip**  
"Replication Dynamics of Gene Vectors Derived from Herpesvirus"  
Master Thesis, 2025.

Please add a reference or acknowledgment in the methods or supplementary material of your study.

License
-------
This code is released for academic use only. Commercial use or redistribution is not permitted
without explicit permission from the author.

Contact
-------
For questions or collaboration inquiries, please contact:
**pfdezvaq@gmail.com**

Enjoy exploring viral dynamics!
