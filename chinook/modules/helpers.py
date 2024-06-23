import matplotlib.pyplot as plt
import pandas as pd

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
