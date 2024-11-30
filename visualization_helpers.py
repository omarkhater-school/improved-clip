import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

def save_and_show_plot(fig, output_dir, filename):
    """
    Saves the plot to a specified directory and shows it.

    Parameters:
        fig (matplotlib.figure.Figure): The figure to save and show.
        output_dir (str): Directory where the figure should be saved.
        filename (str): The name of the file (including extension) to save the figure as.
    """
    os.makedirs(output_dir, exist_ok=True)  # Ensure the output directory exists
    file_path = os.path.join(output_dir, filename)
    fig.savefig(file_path)
    print(f"Figure saved to {file_path}")
    plt.show()  # Display the figure


def visualize_trends(df, x_axis, hue, y_axis='FinalObjectiveValue', title='Trend Analysis', output_dir=".", filename="trend_analysis.png"):
    """
    Visualizes trends in the DataFrame for better insight.

    Parameters:
        df (pd.DataFrame): The input DataFrame containing the data to be visualized.
        x_axis (str): The column to be used for the x-axis (e.g., 'loss_function' or 'optimizer').
        hue (str): The column to use for coloring the categories (e.g., 'optimizer' or 'loss_function').
        y_axis (str): The column to use for the y-axis values. Defaults to 'FinalObjectiveValue'.
        title (str): The title of the plot. Defaults to 'Trend Analysis'.
        output_dir (str): Directory to save the plot. Defaults to current directory.
        filename (str): Name of the file to save the plot. Defaults to 'trend_analysis.png'.
    """
    plt.figure(figsize=(14, 8))
    sns.barplot(data=df, x=x_axis, y=y_axis, hue=hue, dodge=True)
    plt.suptitle(title, fontsize=16)
    plt.xlabel(x_axis.replace('_', ' ').title(), fontsize=12)
    plt.ylabel(y_axis.replace('_', ' ').title(), fontsize=12)
    plt.xticks(rotation=45, fontsize=10)
    plt.legend(title=hue.replace('_', ' ').title(), fontsize=10)
    # plt.tight_layout()
    plt.grid()

    # Save and display the plot
    save_and_show_plot(plt.gcf(), output_dir, filename)


def plot_objective_metric_progress(df, figtitle="Objective Metric Progress", output_dir=".", filename="objective_metric_progress.png"):
    """
    Plots the progress of the objective metric over training experiments.

    Parameters:
        df (pd.DataFrame): DataFrame containing the data.
        figtitle (str): Title of the plot.
        output_dir (str): Directory to save the plot. Defaults to current directory.
        filename (str): Name of the file to save the plot. Defaults to 'objective_metric_progress.png'.
    """
    completed_jobs = df.query("TrainingJobStatus == 'Completed'")
    objective_value_progress = completed_jobs["FinalObjectiveValue"].reset_index(drop=True)

    # Calculate best objective value so far
    best_so_far = objective_value_progress.cummax()

    # Plot results
    plt.figure(figsize=(12, 6))

    # Scatter plot of all completed objective values
    plt.scatter(objective_value_progress.index, objective_value_progress, label="Objective Values")

    # Plot best objective value so far as a solid line
    plt.plot(best_so_far.index, best_so_far, color='red', linestyle='-', label="Best Objective So Far")

    # Enhance the plot
    plt.grid()
    plt.xlabel("Experiment Index")
    plt.ylabel("Objective Metric Value")
    plt.suptitle(figtitle)
    plt.legend()
    # plt.tight_layout()

    # Save and display the plot
    save_and_show_plot(plt.gcf(), output_dir, filename)




def calculate_feature_importance(df, metric_column="FinalObjectiveValue", features=None, model_type="random_forest"):
    """
    Calculates feature importance using a tree-based model.

    Parameters:
        df (pd.DataFrame): DataFrame containing hyperparameters and their objective values.
        metric_column (str): The column name for the objective metric.
        features (list, optional): List of hyperparameters to analyze. If None, all features except the metric_column
                                   and irrelevant columns will be used.
        model_type (str): Type of tree-based model to use ('random_forest'). Defaults to Random Forest.

    Returns:
        pd.DataFrame: A DataFrame containing features and their calculated importances.
    """
    # Select relevant features (exclude the metric column and irrelevant columns)
    if features is None:
        exclude_columns = [metric_column, "TrainingElapsedTimeSeconds", "TrainingJobStatus"]
        features = [col for col in df.columns if col not in exclude_columns]

    X = df[features].copy()
    y = df[metric_column]

    # Encode categorical features
    for col in X.select_dtypes(include=["object", "category"]).columns:
        X[col] = LabelEncoder().fit_transform(X[col])

    # Handle missing values
    X = X.fillna(0)  # Replace missing values in features
    y = y.fillna(y.mean())  # Replace missing target values

    # Train a tree-based model
    if model_type == "random_forest":
        model = RandomForestRegressor(n_estimators=100, random_state=42)
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")

    model.fit(X, y)

    # Extract feature importances
    importances = model.feature_importances_

    # Create a DataFrame of feature importances
    importance_df = pd.DataFrame({
        "Feature": X.columns,
        "Importance (%)": (importances / importances.sum()) * 100  # Normalize to percentage
    }).sort_values(by="Importance (%)", ascending=False)

    return importance_df


def plot_relative_importance(df, 
                             metric_column="FinalObjectiveValue", 
                             features=None, 
                             bar_height=0.2, 
                             importance_threshold=2,
                             figtitle="Relative Importance of Hyperparameters", 
                             output_dir=".", 
                             filename="relative_importance.png"):
    """
    Plots the relative importance of all hyperparameters based on their impact on the objective metric
    as a horizontal bar chart with values displayed on the bars.

    Parameters:
        df (pd.DataFrame): DataFrame containing hyperparameters and their objective values.
        metric_column (str): The column name for the objective metric.
        features (list, optional): List of hyperparameters to analyze. If None, all features except the metric_column
                                   and other irrelevant columns will be used.
        bar_height (float, optional): Height of each bar in the plot (default is 0.5 for clean spacing).
        output_dir (str): Directory to save the plot. Defaults to current directory.
        filename (str): Name of the file to save the plot. Defaults to 'relative_importance.png'.
    """
    # Calculate feature importance
    importance_df = calculate_feature_importance(df, metric_column, features)

    # Split into above and below threshold
    above_threshold = importance_df[importance_df["Importance (%)"] >= importance_threshold]
    below_threshold = importance_df[importance_df["Importance (%)"] < importance_threshold]
    # Add "Others" row for below threshold features
    if not below_threshold.empty:
        others_importance = below_threshold["Importance (%)"].sum()
        others_row = pd.DataFrame({"Feature": ["Others"], "Importance (%)": [others_importance]})
        truncated_features = below_threshold["Feature"].tolist()
        importance_df = pd.concat([above_threshold, others_row], ignore_index=True)
    else:
        truncated_features = []

    # Dynamically adjust the figure height
    num_features = len(above_threshold)
    fig_height = max(5, num_features)  # Minimum height of 3 inches
    fig_width = 10

    # Plot relative importance as a horizontal bar plot
    plt.figure(figsize=(fig_width, fig_height))
    ax = sns.barplot(
        y="Feature", 
        x="Importance (%)", 
        data=importance_df,
        orient="h"
    )
    plt.suptitle(figtitle, fontsize = 16)
    plt.xlabel("Relative Importance (%)")
    plt.ylabel("Hyperparameters")
    plt.grid(axis="x", linestyle="--", alpha=0.7)  # Add gridlines for x-axis

    # Add text annotations at the end of each bar
    for container in ax.containers:  # Iterate through the bar containers
        for bar in container:
            width = bar.get_width()  # Get the bar width (value of Importance (%))
            y = bar.get_y() + bar.get_height() / 2  # Position at the center of the bar height
            ax.text(
                width, 
                y, 
                f"{width:.2f}%",
                ha="left", 
                va="center", 
                color="red", 
                fontsize=13
            )

    # Add a text box for "Others"
    if truncated_features:
        # Create two columns of truncated features
        truncated_text = "\n".join([
            f"{truncated_features[i]:<20}{truncated_features[i+1]:<20}" if i+1 < len(truncated_features) 
            else f"{truncated_features[i]}"
            for i in range(0, len(truncated_features), 2)
        ])

        # Add the text box
        plt.gcf().text(
            .95, 
            0.5,
            f"Features grouped as 'Others':\n{truncated_text}", 
            ha="left", 
            va="center", 
            bbox=dict(boxstyle="round", facecolor="lightgrey", edgecolor="black")
        )

    # Save and display the plot
    save_and_show_plot(plt.gcf(), output_dir, filename)






def visualize_phase(df, filtered_df, phase_number, output_dir="."):
    """
    Visualizes trends and metrics for a given phase.

    Parameters:
        df (pd.DataFrame): The full DataFrame containing all data.
        filtered_df (pd.DataFrame): The filtered DataFrame containing data for the specific phase.
        phase_number (int): The phase number.
        output_dir (str): Directory to save the plots. Defaults to current directory.
    """
    # Create phase-specific subfolder
    phase_dir = os.path.join(output_dir, f"phase_{phase_number}")
    os.makedirs(phase_dir, exist_ok=True)

    # Fig1: Final Objective Value by Loss Function with Optimizer as Dimension
    if phase_number < 3:
        visualize_trends(
            filtered_df,
            x_axis='loss_function',
            hue='optimizer',
            title=f'Final Objective Value by Loss Function with Optimizer as Dimension - Phase {phase_number}',
            output_dir=phase_dir,
            filename=f"loss_function_vs_optimizer_phase_{phase_number}_dim_opt.png"
        )
        visualize_trends(
            filtered_df,
            x_axis='optimizer',
            hue='loss_function',
            title=f'Final Objective Value by Optimizer with Loss Function as Dimension - Phase {phase_number}',
            output_dir=phase_dir,
            filename=f"loss_function_vs_optimizer_phase_{phase_number}_dim_loss.png"
        )
       
    # Fig2: Objective Metric Progress
    plot_objective_metric_progress(
        df,
        figtitle=f"Objective Metric Progress - Phase {phase_number}",
        output_dir=phase_dir,
        filename=f"objective_metric_progress_phase_{phase_number}.png"
    )

    # Fig3: Variable Importance
    plot_relative_importance(
        df=filtered_df,
        figtitle=f"Relative Importance of Hyperparameters - Phase {phase_number}",
        output_dir=phase_dir,
        filename=f"relative_importance_phase_{phase_number}.png"
    )


def plot_metrics_with_best_epoch(
    metric_dataframes, 
    metrics_to_plot=["ValidationTxtR1", "ValidationImgR1", "ValidationZS1"], 
    objective_metric="ObjectiveValue",
    best_epoch_line=True,
    figsize=(15, 5)
):
    """
    Plot multiple metrics with an indication of the best epoch.

    Parameters:
    - metric_dataframes (dict): Dictionary of DataFrames for each metric.
    - metrics_to_plot (list): List of metrics to plot.
    - objective_metric (str): Name of the objective metric.
    - best_epoch_line (bool): Whether to add vertical and horizontal lines for the best epoch.
    - figsize (tuple): Size of the figure.
    """
    plt.figure(figsize=figsize)

    # Plot each metric with dashed lines
    for metric in metrics_to_plot:
        plt.plot(
            metric_dataframes[metric].index,
            metric_dataframes[metric]['value'],
            label=metric,
            linestyle='--'
        )

    # Plot the objective metric with a solid line
    plt.plot(
        metric_dataframes[objective_metric].index,
        metric_dataframes[objective_metric]['value'],
        label=f"{objective_metric} (Average)",
        linewidth=2
    )

    # Add vertical and horizontal lines for the best epoch
    if best_epoch_line:
        # Find the best epoch based on the maximum value of the objective metric
        best_epoch_idx = metric_dataframes[objective_metric]['value'].idxmax()
        best_epoch_value = metric_dataframes[objective_metric].loc[best_epoch_idx, 'value']

        # Add vertical line for the best epoch
        plt.axvline(
            x=best_epoch_idx, 
            color='red', 
            linestyle='-.', 
            label=f'Best Epoch: {best_epoch_idx}'
        )

        # Add horizontal line for the best objective value
        plt.axhline(
            y=best_epoch_value, 
            color='green', 
            linestyle='-.', 
            label=f'Best Value: {best_epoch_value:.2f}'
        )

    # Add labels, title, and grid
    plt.xlabel('Epoch')
    plt.ylabel('Metric Value')
    plt.title('Validation Metrics During Training')
    plt.grid(True)

    # Add legend outside the plot area
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

    # Adjust layout to make space for the legend
    plt.tight_layout()

    # Show the plot
    plt.show()
