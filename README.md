# Water Index Futures
This repo contains code and data for the following paper:
Dan Li, Gupta, R, Zeff. H.B., & Characklis, G.W.
Managing Drought Related Financial Risks with Water Futures. (In preparation)

**Abstract**
>The recently launched California water price index (Nasdaq: NQH2O) and the corresponding water futures (CME: H2O) offer water users, such as industrial farms, irrigation districts, and municipalities, an opportunity to hedge against unfavorable price movements in California's water market, particularly during extended drought periods. In this study, we explore the potential of hydrologic forecasting to predict water index prices using ensemble forecasts of daily time-series of water delivery, storage levels, and irrigation district demand in the Central Valley of California. Forecasting is conducted with a random forest model, trained on hydrologic data from 2013 to 2024, which achieves an R² value of 0.98. These predictions are then applied to a 9-month futures contract to manage water costs for an industrial farm in California. With 50% of water demand covered by futures contracts, the maximum cost is reduced by 12.9%, and the variance across all scenarios is reduced by 14.5%. The seasonal hydrologic forecasting model of water prices, therefore, provides a hedging tool to reduce water purchase cost volatility with relatively low fees, addressing the critical challenges posed by constrained water resources and competing demands.

# California Food-Energy-Water System (CALFEWS)
For general information on the California Food-Energy-Water System (CALFEWS) simulation model please refer to main branch of (https://github.com/hbz5000/CALFEWS).
Please refer to the following paper to learn more about the performance and conceptual underpinnings of the model:

Zeff, H.B., Hamilton, A.L., Malek, K., Herman, J.D., Cohen, J.S., Medellin-Azuara, J., Reed, P.M., and G.W. Characklis. (2021). California's Food-Energy-Water System: An Open Source Simulation Model of Adaptive Surface and Groundwater Management in the Central Valley. *Environmental Modelling & Software, 141*: 105052. [https://doi.org/10.1016/j.envsoft.2021.105052](https://doi.org/10.1016/j.envsoft.2021.105052) 

Download the exact version used to produce the paper at [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.4091708.svg)](https://doi.org/10.5281/zenodo.4091708).

Licensed under the MIT License, 2017.

## Installation and setup
1. First, switch to the "main" branch of CALFEWS [https://github.com/hbz5000/CALFEWS] and follow instructions to clone, compile, and run the model CALFEWS_waterFutures in the same approach.
2. Ensure you cythonize and recompile the code whenever changes are made to the .pyx files by using the following command:  ``python setup_cy.py build_ext --inplace`` .
3. Navigate to ``CALFEWS_intro_tutorial.ipynb`` to run the validation/simulation. Before proceeding, ensure that you configure the runtime_params.ini file appropriately. For running synthetic simulations, set model_run = simulation and flow_input_type = synthetic. Note that completing 100 synthetic runs takes approximately 2-3 hours. It is advisable to start with a small batch trial to become familiar with the process. 
4. Once the synthetic run is complete, the results will be stored in the "results" folder. Each output is saved in a subfolder named 1 to 100. After this, execute the  ``ensemble_pred.ipynb `` notebook to perform water index forecasting.

