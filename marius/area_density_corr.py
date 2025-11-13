import matplotlib.pyplot as plt
import seaborn as sns

def visualize_categorical_relationships(df, numeric_col='Density', categorical_cols=['Area', 'Region']):
    """
    Creates visualizations to show relationship between numeric and categorical variables.
    """
    n_plots = len(categorical_cols)
    fig, axes = plt.subplots(1, n_plots, figsize=(7 * n_plots, 5))

    if n_plots == 1:
        axes = [axes]

    for idx, cat_col in enumerate(categorical_cols):
        # Box plot to show distribution
        df_sorted = df.sort_values(cat_col)
        sns.boxplot(data=df_sorted, x=cat_col, y=numeric_col, ax=axes[idx])
        axes[idx].set_title(f'{numeric_col} by {cat_col}', fontsize=12, fontweight='bold')
        axes[idx].set_xlabel(cat_col, fontsize=10)
        axes[idx].set_ylabel(numeric_col, fontsize=10)
        axes[idx].tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.show()

    # Also create violin plots for more detail
    fig, axes = plt.subplots(1, n_plots, figsize=(7 * n_plots, 5))

    if n_plots == 1:
        axes = [axes]

    for idx, cat_col in enumerate(categorical_cols):
        df_sorted = df.sort_values(cat_col)
        sns.violinplot(data=df_sorted, x=cat_col, y=numeric_col, ax=axes[idx])
        axes[idx].set_title(f'{numeric_col} Distribution by {cat_col}', fontsize=12, fontweight='bold')
        axes[idx].set_xlabel(cat_col, fontsize=10)
        axes[idx].set_ylabel(numeric_col, fontsize=10)
        axes[idx].tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.show()

# visualize_categorical_relationships(df, numeric_col='Density', categorical_cols=['Area', 'Region'])