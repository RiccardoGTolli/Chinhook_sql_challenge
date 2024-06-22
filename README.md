# Chinhook_sql_challenge - Riccardo














The files for the first and second task are all contained in revo/notebooks.
To get access you should run the container in development mode i.e. in the docker-compose.yml, you should have target: development (default).

The third task consists in the Docker app itself and all the files in revo/modules.
You can inspect and execute the files in development mode just fine but if you want the docker container to auto-execute the main.py you should run the container with target: production.

## Run the container
In the same directory where the docker-compose.yml is, open a linux terminal, on Windows you can use WSL(Windows Subsystem for Linux) and run:

```bash
docker compose up --build -d
```

## Execute Jupyter cells
Jupyter is exposed on port 8888, to access the service you need the token.

How to get the token:
```bash
docker compose exec revo jupyter server list
```
The token will change everytime the container is restarted.

method 1: Access Jupyter in browser
- Go to localhost:8888
- Type in the token

method 2: Within the IDE e.g. VsCode
- In a .ipynb file select the Kernel in the top right
- Select Existing Jupyter Server
- Type in the URL of the Jupyter Server which will be:
  http://localhost:8888/?token=<your_token>
- Type a name, the default 'localhost' is fine
- Select Python 3 (ipykernel)
  
## Gain access to container shell
Once the container is running (with docker compose up) you can gain access to the container shell with:

```bash
docker compose exec revo bash
```

## Run the app from start to finish

Method 1:
Build and run the image in production mode i.e., in the docker-compose.yml, have: 'target: production'.
Dont use the flag -d so that you can see the logs.
```bash
docker compose up --build
```
Once the image is built, you can just `docker compose up` to re-execute the app.
For example if you change a parameter in `config.json`, that is part of a volume, so you dont need to rebuild the app.


Method 2: Gain access to container shell and execute main.py manually.
With 'target: development': 
```bash
docker compose up --build -d
docker compose exec revo bash
```
Then execute main.py manually:
```bash
python3 modules/main.py --config modules/config.json
```
You can play with the arguments in config.json to obtain different model accuracy and change the start and end period. Simply change the values in config.json and rerun the app.
## Formatting, linters and static-checks (just for development purposes)

From inside container, run `format` and `code-checks`.

## Unit Testing

This has been done just for illustrative purposes, only a single function was unit tested: `ml_helpers.get_year_quarter_combos`.
You can run the test like so:
1. Gain access to the container CLI: `docker compose exec revo bash`
2. Then execute the test with : `python3 modules/test_ml_helpers.py`

## Description of the app

<div style="background-color: #1a1c1f; padding: 10px;">

#### The notebooks and the modules do the same thing, except that the modules (main.py) has a few extra features:
1. `config.json`, this allows you to change the parameters to obtain different results for example hyperparameter selection for improved performance or change the start and end period.
Look at `sample_config.json` to understand what kind of parameters can be put in the `config.json`.
2. For task 2 the notebooks simply print the correlation matrix while the modules save the best result matrix to `output/4_task2`.
3. The modules allow you to run the model with or without Missing Indicator columns (via `mi_cols` parameter in `config.json`) while the notebooks use them by default.
 </div>

&nbsp;

<div style="background-color: #1a1c1f; padding: 10px;">

#### How the tasks were tackled:

1.  Cleaning the data:
    1. The data is stored in .arff files, this severely limits the possibilities on python to use well established libraries such as pandas, therefore a few ad-hoc cleaning steps have been applied to clean the files by treating them as text files.
    2. After cleaning, the .arff files are put into a single pandas DataFrame and we do common cleaning procedures such as handle missings, encode variables, remove outliers, missing indicator and inputation.
    3. The file that does this is clean_data.py, it creates `'output/0_cleaned_data/df_task1.csv'` and  `'output/0_cleaned_data/df_task2.csv'`; the difference between the two files is that the latter is used for a classification problem so it s lacking some preprocessing steps that will be applied to train and test separately to avoid data leakage.

2. Exploratory Analysis:
   1. The file that does this step is `task1_exploratory_analysis.py`.
   2. The output is simply two plots, the first is a time series line plot grouped by country, saved in `'output/1_task1_exploratory_analysis/Line plot grouped by country.png'`; the second plot is a time series line plot grouped by sector, saved in `'output/1_task1_exploratory_analysis/Line plot grouped by sector.png'`.

3. Find the financial indicators that statistically changed between 2019 Q4 and 2020 Q2:
   1. The timeframe can actually be decided before any run by simply using these fields in the config.json: `start_year`, `start_quarter`, `end_year`, `end_quarter`.
   2. We run a linear regression using Year and Quarter as x variables and a single financial indicator as y variable; therefore we run a model for each financial indicator.
   3. Quarter has been trigonometrically encoded to preserve its cyclical nature, therefore it has been split in `sin_quarter` and `cos_quarter`.
   4. Using the average of the coefficients and pvalues for Year and Quarter we determine the indicators whose change is statistically significant.
   5. The financial indicators that satisfy the pvalue requirement (which can be changed in config) are saved in `'output/2_task1_a/df_task1_a_result.csv'` , you will also find plots for each feature in `'output/2_task1_a/'`

4. Find, for each sector, the financial indicators that statistically changed between 2019 Q4 and 2020 Q2. Rank these financial indicators based on the number of sectors in which they significantly changed:
   1. This is the same concept as task1_a but we are grouping the data by sector.
   2. You can find a ranking of statistically significant indicators by Sector in `'output/3_task1_b/df_task1_b_result.csv'`; while the plots are in `'output/3_task1_b/'`

5. Implement a classification machine learning algorithm to predict the sector of the companies:
   1. Data is prepared for supervised learning, split in train and test,  three models are applied on it: Random Forest, LightGBM and XGBoost.
   2. Hyperparameters can be changed in the `config.json`, look at `sample_config.json` to get ideas on which parameters to use.
   3. The best model's correlation matrix will be output in `'output/4_task2'`, best model is chosen according to the metric Accuracy.
