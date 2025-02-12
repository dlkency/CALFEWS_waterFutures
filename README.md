# Water Index Futures
This repo contains code and data for the following paper:
Dan Li, Gupta, R, Zeff. H.B., & Characklis, G.W.
Managing Drought Related Financial Risks with Water Futures. (In preparation)

**Abstract**
>The recently launched California water price index (Nasdaq: NQH2O) and the corresponding water futures (CME: H2O) offer water users, such as industrial farms, irrigation districts, and municipalities, an opportunity to hedge against unfavorable price movements in California's water market, particularly during extended drought periods. In this study, we explore the potential of hydrologic forecasting to predict water index prices using ensemble forecasts of daily time-series of water delivery, storage levels, and irrigation district demand in the Central Valley of California. Forecasting is conducted with a random forest model, trained on hydrologic data from 2013 to 2024, which achieves an R² value of 0.98. These predictions are then applied to a 9-month futures contract to manage water costs for an industrial farm in California. With 50% of water demand covered by futures contracts, the maximum cost is reduced by 12.9%, and the variance across all scenarios is reduced by 14.5%. The seasonal hydrologic forecasting model of water prices, therefore, provides a hedging tool to reduce water purchase cost volatility with relatively low fees, addressing the critical challenges posed by constrained water resources and competing demands.

# California Food-Energy-Water System (CALFEWS)
For general information on the California Food-Energy-Water System (CALFEWS) simulation model please refer to main branch of https://github.com/dlkency/CALFEWS/blob.
Please refer to the following paper to learn more about the performance and conceptual underpinnings of the model:

Zeff, H.B., Hamilton, A.L., Malek, K., Herman, J.D., Cohen, J.S., Medellin-Azuara, J., Reed, P.M., and G.W. Characklis. (2021). California's Food-Energy-Water System: An Open Source Simulation Model of Adaptive Surface and Groundwater Management in the Central Valley. *Environmental Modelling & Software, 141*: 105052. [https://doi.org/10.1016/j.envsoft.2021.105052](https://doi.org/10.1016/j.envsoft.2021.105052) 

Download the exact version used to produce the paper at [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.4091708.svg)](https://doi.org/10.5281/zenodo.4091708).

Licensed under the MIT License, 2017.

## Installation and setup
1. Clone this repository to your local machine.
1. If you use Anaconda:
    1. Create a new environment using the yml file: ``conda env create -f environment.yml``
    1. Activate environment: ``conda activate .venv_conda_calfews``
1. If you don't use Anaconda:
    1. Manually install the packages listed in ``environment.yml`` into a new virtual environment named ``.venv_conda_calfews``, and activate the environment.
1. From the base CALFEWS directory, run model with ``python -W ignore run_main_cy.py <results_folder>``, where ``<results_folder>`` is the location you would like to store the results, relative to base directory. (Note: the command for Python 3 may be python3, not python, depending on your machine).
1. If this doesn't work (or you want to make any changes to source files), you will need to recreate the C files & binaries from Cython. 
    1. If you are running on Linux or MacOS, you should already have gcc installed. If you are running on Windows, you will need to install Visual Studio 2019 Community Edition. When it asks which programs to install, choose "Desktop development with C++".
    1. Cythonize and recompile with the command: ``python setup_cy.py build_ext --inplace``.
1. Three different Jupyter notebooks are available to improve usability. To use them, you will need to run the following commands in the terminal or Anaconda Prompt.
    1. Add your virtual environment to Jupyter: ``python -m ipykernel install --user --name=.venv_conda_calfews``
    1. Tell Jupyter to allow ipywidgets: ``jupyter nbextension enable --py widgetsnbextension`` (this step may or may not be necessary depending on your Jupyter version)
    1. Tell Jupyter to allow bqplot: ``jupyter nbextension enable --py bqplot``
1. The three helper notebooks are:
    1. ``CALFEWS_GUI.ipynb``: This graphical user interface (GUI) allows the user to run different inflow scenarios and visualize results interactively (run the notebook and scroll down to the bottom for the GUI.
    1. ``CALFEWS_intro_tutorial.ipynb``: This notebook gives more details on model parameters, input files, and output files, for users who want to go deeper than the GUI.
    1. ``modeling_paper_notebook.ipynb``: This notebook can be run to automatically run run all simulations and reproduce all figures from Zeff et al. 2020 (see citation above).
