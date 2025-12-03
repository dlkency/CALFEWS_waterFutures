# Water Index Futures
This repo contains code and data for the following paper:
Managing Drought Related Financial Risks with Water Futures. 

**Abstract**
>The recently launched California water price index (Nasdaq: NQH2O) and the corresponding water futures (CME: H2O) offer water users, such as industrial farms, irrigation districts, and municipalities, an opportunity to hedge against unfavorable price movements in California's water market, particularly during extended drought periods. In this study, we explore the potential of hydrologic forecasting to predict water index prices using ensemble forecasts of daily time-series of water delivery, storage levels, and irrigation district demand in the Central Valley of California. Forecasting is conducted with a random forest model, trained on hydrologic data from 2013 to 2024. These predictions are then applied to a 1-, 3-, 6-month futures contract to manage water costs for an industrial farm in California. With 50% of water demand covered by futures contracts, the maximum cost is reduced by 12.9%, and the variance across all scenarios is reduced by 14.5%. The seasonal hydrologic forecasting model of water prices, therefore, provides a hedging tool to reduce water purchase cost volatility with relatively low fees, addressing the critical challenges posed by constrained water resources and competing demands.

# California Food-Energy-Water System (CALFEWS)
For general information on the California Food-Energy-Water System (CALFEWS) simulation model please refer to (https://github.com/hbz5000/CALFEWS).
Please refer to the following paper to learn more about the performance and conceptual underpinnings of the model:
Zeff, H.B., Hamilton, A.L., Malek, K., Herman, J.D., Cohen, J.S., Medellin-Azuara, J., Reed, P.M., and G.W. Characklis. (2021). California's Food-Energy-Water System: An Open Source Simulation Model of Adaptive Surface and Groundwater Management in the Central Valley. *Environmental Modelling & Software, 141*: 105052. [https://doi.org/10.1016/j.envsoft.2021.105052](https://doi.org/10.1016/j.envsoft.2021.105052) 
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
3. Navigate to ``CALFEWS_intro_tutorial.ipynb`` to run the validation/simulation. Before proceeding, ensure that you configure the runtime_params.ini file appropriately. For running synthetic simulations, set model_run = simulation and flow_input_type = synthetic.  

## Climate Emsemble
1. Generate the ensemble on an HPC cluster. Using the parallel job structure defined in climate_ensemble/submit_100jobs_ensemble.sh, the full ensemble completes in 20–30 minutes. Running the equivalent workload locally would take approximately 60 hours.
2. Once the ensemble run is done, make sure input data is in calfews_src/data/input and enesemble results are saved in results/startyear_4_1 to proceed.
3. Run hybrid_prediction_decay.py to train the machine-learning models used to predict water-index prices from lagged price and hydrologic features.
4. Run ensemble_water_price_uncert.py to generate price estimates for each ensemble trace, incorporating predictive uncertainty.
5. Run water_hedging_strategy.py to construct the hedging strategy and evaluate its effectiveness across the full ensemble.
