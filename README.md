# chinook_sql_challenge - Riccardo














The tasks have been carried out in chinook/nooteboks, which import functions present in chinook/modules.
If you want to run the notebooks you should run the container in development mode i.e. in the docker-compose.yml, you should have target: development (default).

The production entry script is not supported at the moment as there is no deployment.

## Run the container
In the same directory where the docker-compose.yml is, open a linux terminal, on Windows you can use WSL(Windows Subsystem for Linux) and run:

```bash
docker compose up --build -d
```

## Execute Jupyter cells
Jupyter is exposed on port 8888, to access the service you need the token.

How to get the token:
```bash
docker compose exec chinook jupyter server list
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
docker compose exec chinook bash
```

## Formatting, linters and static-checks (just for development purposes)

From inside container, run `format` and `code-checks`.

## Description of the app
There is a Python main container and a Postgres container with the data. You can check docker-compose.yml for the details.
<div style="background-color: #1a1c1f; padding: 10px;">

#### How the tasks were tackled:

1.  Task A:
    
    The 4 questions in task A were answered by creating a view for each question in the database service.
    The views are created in chinook_db/Chinook_PostgreSql.sql.
    They are also imported in notebooks/task_a.ipynb, alternatively you can connect to the database service with some db management tool like DBeaver and inspect the database directly.
    The db connection info can be found in the .env file.

2. Task B:
   
   This was achieved by simply creating a consolidated view in the database.
   The code can be found in chinook_db/Chinook_PostgreSql.sql.
   The notebooks/task_b.ipynb imports the table and prints it.

3. Task C:
   1. In notebooks/task_c_1.ipynb you can find some exploratory analysis, carried out by plotting some Pareto Charts, this will show the top and cumulative revenue yielders by customer_id, country, genre_name, media_type and artist_name.
   2. In notebooks/task_c_2.ipynb a few different analysis have been carried out:
      
      1. Feature Extraction: We have calculated some extra columns that can aid with analysis, these columns are : 
         - Number of transactions per customer
         - Total Number of Tracks Purchased per customer
         - Total Spent per Customer ID
         - Average spend per customer
         - Favorite Genre
         - Frequency of Purchase
         - Recency (days between 2026-01-01 and last purchase)
      2. K-Means clustering:
         - Preprocess
         - Using a plot for determining the optimal number of clusters with the elbow method
         - Total Spent per Customer ID
         - Assign the clusters to the customers
         - Create a cluster_segments dataframe where the data is grouped by Cluster and aggregated my mean and mode. This is helpful to individuate the characteristics of the cluster.
         - Plot the clusters with PCA (for dimensionality reduction)
      3. Predictive Customer Lifetime Value (CLV):
         This is just an illustrative part that showcases possible next steps.
         We predict total_spent from the entire dataset, the point is that we can use total_spent as proxy for CLV.
         In order for this step to be complete we would need to feature extract CLV and use it as a y variable.
         - We use an xgboost algorithm
         - We use SHAP to show which features are the most important

