import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import numpy as np
from scipy.stats import mode
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error
import shap

def pareto_chart(df, values:str, grouping:str, title:str, topn:int=None):
    '''Shows to console a pareto chart using the values and grouping.
    `values` and `grouping` are two columns from the df.
    `topn` limits the number of top entries to display.
    '''
    # Aggregate data by the grouping column and sum up the values
    df_agg = df.groupby(grouping)[values].sum().reset_index()

    # Sort the DataFrame by aggregated values in descending order
    df_agg = df_agg.sort_values(by=values, ascending=False)

    # If topn is specified, limit the DataFrame to top n rows
    if topn is not None:
        df_agg = df_agg.head(topn)

    # Reset index to keep track of the order after sorting
    df_agg.reset_index(drop=True, inplace=True)

    # Calculate cumulative sum of values and then the cumulative percentage
    df_agg['cumulative_sum'] = df_agg[values].cumsum()
    df_agg['cumulative_percentage'] = 100 * df_agg['cumulative_sum'] / df_agg[values].sum()

    fig, ax1 = plt.subplots(figsize=(10, 5))

    # Bar plot for values using the new index for x and labels for grouping
    ax1.bar(df_agg.index, df_agg[values], color='skyblue', label=values)
    ax1.set_xlabel(grouping)
    ax1.set_ylabel(values)
    ax1.set_xticks(df_agg.index)  # Set x-ticks to position based on DataFrame index
    ax1.set_xticklabels(df_agg[grouping], rotation=90)  # Set x-tick labels from the grouping column
    ax1.set_title(title)

    # Line plot for cumulative percentage
    ax2 = ax1.twinx()
    ax2.plot(df_agg.index, df_agg['cumulative_percentage'], color='C1', marker='', linestyle='-', linewidth=2, label='Cumulative %')
    ax2.set_ylabel('Cumulative Percentage (%)')
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, loc: "{:.0f}%".format(int(x))))

    # Adding a legend to the chart
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc='upper left')

    plt.show()
    plt.clf()  # Clears the figure


def feature_extraction(df:pd.DataFrame):
    '''This function adds some calculated columns that will aid analysis.'''
    # Number of transactions per customer
    df['transactions_per_customer'] = df.groupby('customer_id')['customer_id'].transform('count')
    
    # Total Number of Tracks Purchased per customer
    df['total_tracks_purchased'] = df.groupby('customer_id')['tracks_purchased'].transform('sum')
    
    # Total Spent per Customer ID
    df['total_spent_per_customer'] = df.groupby('customer_id')['total_spent'].transform('sum')
    
    # Average spend per customer
    df['average_spend'] = df.groupby('customer_id')['total_spent'].transform('mean')
    
    # Favorite Genre
    # This creates a temporary DataFrame counting genres per customer
    genre_mode = df.groupby(['customer_id', 'genre_name']).size().reset_index(name='counts')
    # This filters the rows by the maximum count per customer
    genre_mode = genre_mode.loc[genre_mode.groupby('customer_id')['counts'].idxmax()]
    df = pd.merge(df, genre_mode[['customer_id', 'genre_name']], on='customer_id', how='left')
    df.rename(columns={'genre_name_y': 'favorite_genre', 'genre_name_x': 'genre'}, inplace=True)
    
    # Frequency of Purchase
    df['invoice_date'] = pd.to_datetime(df['invoice_date'], format='%Y-%m-%d')
    # Calculate the span of days between the first and last purchase
    df['first_purchase'] = df.groupby('customer_id')['invoice_date'].transform('min')
    df['last_purchase'] = df.groupby('customer_id')['invoice_date'].transform('max')
    df['days_active'] = (df['last_purchase'] - df['first_purchase']).dt.days
    
    # Calculate purchase frequency . Add 1 to avoid division by zero
    df['purchase_frequency'] = df['transactions_per_customer'] / (df['days_active'] + 1)  

    # Recency (days between 2026-01-01 and last purchase). 
    # I selected that date because some purchases have been made in 2025
    df['recency'] = (pd.Timestamp('2026-01-01') - df['last_purchase']).dt.days

    df.drop(['first_purchase','last_purchase'],axis=1,inplace=True)
    
    return df


def summarize_data(df):

    df = df.drop(columns=['invoice_date'], errors='ignore')
    # Define categorical and numerical features
    categorical_features = ['country', 'city', 'state', 'company', 'media_type', 'genre', 
                            'composer', 'album_title', 'artist_name', 'favorite_genre']
    features_sum = ['tracks_purchased', 'total_spent']
    features_mean = ['transactions_per_customer', 'total_tracks_purchased', 
                     'total_spent_per_customer', 'average_spend', 
                     'days_active', 'purchase_frequency', 'recency']
    
    # Grouping by 'customer_id'
    # For categorical features, we calculate the mode (most frequent occurrence)
    categorical_summary = df.groupby('customer_id')[categorical_features].agg(lambda x: x.mode().iat[0] if not x.mode().empty else None)
    
    # Group by 'customer_id' and aggregate with sum for specific features
    numerical_summary_sum = df.groupby('customer_id')[features_sum].sum()

    # Group by 'customer_id' and aggregate with mean for the other set of features
    numerical_summary_mean = df.groupby('customer_id')[features_mean].mean()

    # Combining summaries into a single DataFrame
    combined_summary = pd.concat([categorical_summary, numerical_summary_sum,numerical_summary_mean], axis=1)

    return combined_summary


def kmeans_preprocessing(df):
    
    # Define categorical and numerical features
    categorical_features = ['country', 'city', 'state', 'company', 'media_type', 'genre', 
                            'composer', 'album_title', 'artist_name', 'favorite_genre']
    numerical_features = ['tracks_purchased', 'total_spent', 'transactions_per_customer', 
                          'total_tracks_purchased', 'total_spent_per_customer', 'average_spend',
                          'days_active', 'purchase_frequency', 'recency']
    
    # Scale numerical features
    scaler = StandardScaler()
    scaled_numerical = scaler.fit_transform(df[numerical_features])
    
    # Encode categorical features
    encoder = OneHotEncoder()
    encoded_categorical = encoder.fit_transform(df[categorical_features]).toarray()
    
    # Get feature names for the encoded categorical data
    cat_feature_names = encoder.get_feature_names_out(categorical_features)
    
    # Combine all feature names
    all_features = numerical_features + list(cat_feature_names)
    
    # Concatenate scaled numerical and encoded categorical features
    X_processed = np.hstack((scaled_numerical, encoded_categorical))
    
    # Convert the processed array back into a DataFrame
    X_df = pd.DataFrame(X_processed, columns=all_features)
    
    return X_df
    

def kmeans_clustering_elbow(df):

    # Determining the number of clusters (k) using the Elbow Method
    sse = []
    for k in range(1, 11):  # Test for 1 to 10 clusters
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(df)
        sse.append(kmeans.inertia_)  # Sum of squared distances of samples to their closest cluster center

    # Plot SSE for each *k*
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, 11), sse, marker='o')
    plt.title('Elbow Method to Determine Optimal Number of Clusters')
    plt.xlabel('Number of Clusters')
    plt.ylabel('SSE (Sum of Squared Errors)')
    plt.show()
    plt.clf()  # Clears the figure
    
def kmeans_clustering(df,clusters=5):
    kmeans = KMeans(n_clusters=clusters, random_state=42)
    cluster_labels=kmeans.fit_predict(df)
    return cluster_labels

def plot_clusters(df, cluster_labels):
    # Reduce dimensions to 2D for visualization using PCA
    pca = PCA(n_components=2)
    principal_components = pca.fit_transform(df)
    
    # Create a DataFrame for the PCA results
    pca_df = pd.DataFrame(data=principal_components, columns=['principal component 1', 'principal component 2'])
    pca_df['Cluster'] = cluster_labels  # Add the cluster labels to the DataFrame
    
    # Plotting
    fig, ax = plt.subplots(figsize=(10, 8))
    colors = ['r', 'g', 'b', 'y', 'c', 'm', 'k', 'orange', 'purple', 'brown']  # Colors for the clusters
    for i in range(max(cluster_labels)+1):
        points = pca_df[pca_df['Cluster'] == i]
        ax.scatter(points['principal component 1'], points['principal component 2'], s=50, c=colors[i], label=f'Cluster {i}')
    
    plt.title('2D PCA Plot of Clusters')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.legend()
    plt.grid(True)
    plt.show()
    plt.clf()  # Clears the figure

def get_mode(series):
    # This function returns the first mode if there are multiple modes.
    m = mode(series)
    if m.count[0] > 0:
        return m.mode[0]
    else:
        return "N/A"  # or some other value that indicates no mode found

def analyze_clusters(df, cluster_labels):
    # Assign cluster labels to your DataFrame
    df['Cluster'] = cluster_labels
    
    # Define categorical and numerical features
    categorical_features = ['country', 'city', 'state', 'company', 'media_type', 'genre', 
                            'composer', 'album_title', 'artist_name', 'favorite_genre']
    numerical_features = ['tracks_purchased', 'total_spent', 'transactions_per_customer', 
                          'total_tracks_purchased', 'total_spent_per_customer', 'average_spend',
                          'days_active', 'purchase_frequency', 'recency']

    # Ensure the correct data types: sometimes, numeric data might be interpreted as object type
    df[numerical_features] = df[numerical_features].apply(pd.to_numeric, errors='coerce')

    # Group by 'Cluster' and calculate mean for numerical features
    cluster_means = df.groupby('Cluster')[numerical_features].mean()

    # Group by 'Cluster' and calculate mode for categorical features using Pandas' mode
    cluster_modes = df.groupby('Cluster')[categorical_features].agg(lambda x: x.mode().iat[0] if not x.mode().empty else "N/A")

    # Combine means and modes into a single DataFrame
    cluster_summary = pd.concat([cluster_means, cluster_modes], axis=1)

    return cluster_summary



def predict_total_spent(df):
    """
    Predicts the 'total_spent' for the entire DataFrame using XGBoost and appends the predictions
    as a new column. It encodes categorical variables, trains the model on part of the data,
    and then applies the model to the entire dataset. Additionally, it calculates and appends the MAE.

    Parameters:
    - df (pd.DataFrame): The input DataFrame with all necessary features and 'total_spent' column.

    Returns:
    - df_final (pd.DataFrame): The original DataFrame with additional columns 'predicted_total_spent'
                               and a constant column 'MAE' showing the mean absolute error.
    """
    # Identify categorical columns
    categorical_features = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # One-hot encode categorical features directly using pandas
    df_encoded = pd.get_dummies(df, columns=categorical_features, drop_first=True)
    
    # Sanitize column names to be compatible with XGBoost
    sanitized_columns = sanitize_column_names(df_encoded.columns)
    df_encoded.columns = sanitized_columns
    
    # Split the data into features and target variable
    X = df_encoded.drop('total_spent', axis=1)
    y = df_encoded[['total_spent']]


    # We dont need to split into train and test because we are using total_spent as proxy for CLV (Customer Lifetime Value) 
    # Lets apply the model on the entire dataset
    # Initialize the XGBoost regressor
    model = XGBRegressor(objective='reg:squarederror', n_estimators=100, learning_rate=0.1, random_state=42)
    model.fit(X, y)
    # Predict on entire dataset
    predictions = model.predict(X)
    # Calculate MAE
    mae = mean_absolute_error(y, predictions)
    # Merge the predictions and MAE back to the original dataframe to retain original categorical data
    df['predicted_total_spent'] = predictions
    df['MAE'] = mae
    
    return df, model, X


def sanitize_column_names(columns):
    """ Sanitize column names to make them compatible with XGBoost. """
    return [col.replace('[', '_').replace(']', '_').replace('<', '_') for col in columns]


def plot_predicted_vs_actual(actual, predicted):
    """
    Plots predicted values against actual values with a diagonal line that represents perfect predictions.

    Parameters:
    - actual (array-like): Array of actual values.
    - predicted (array-like): Array of predicted values.
    """
    plt.figure(figsize=(10, 8))
    plt.scatter(predicted, actual, alpha=0.5, color='blue', label='Predicted vs. Actual')  # Scatter plot
    plt.title('Predicted vs Actual Values')
    plt.xlabel('Predicted Values')
    plt.ylabel('Actual Values')

    # Draw a diagonal line (perfect predictions line)
    max_val = max(max(actual), max(predicted))
    min_val = min(min(actual), min(predicted))
    plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', label='Line of Perfect Prediction')

    plt.legend()
    plt.grid(True)
    plt.show()
    plt.clf()  # Clears the figure
    
def plot_shap_absmean(model,X):
    X=X.astype('float')
    # Compute SHAP values
    explainer = shap.Explainer(model, X)
    shap_values = explainer(X)

    # SHAP Summary Plot (Mean SHAP values)
    shap.summary_plot(shap_values, X, plot_type="bar")
    plt.clf()  # Clears the figure