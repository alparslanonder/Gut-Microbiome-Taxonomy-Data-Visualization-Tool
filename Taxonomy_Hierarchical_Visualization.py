import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import os

def process_xlsx(sheet_data,start_rank,taxonomy,threshold):
    # Ensure 'RankID' column is string type (if needed)
    sheet_data['RankID'] = sheet_data['RankID'].astype(str)

    # Filter and combine data for all ranks from start_rank to start_rank + taxonomy
    rank_mask = sheet_data['RankID'].str.count(r'\.').between(start_rank, start_rank + taxonomy - 1)
    filtered_data_combined = sheet_data[rank_mask]

    # Extract and aggregate data by 'Taxon' for columns ending in 'A' (After) and 'B' (Before)
    before_columns = [col for col in filtered_data_combined.columns if col.endswith('B')]
    after_columns = [col for col in filtered_data_combined.columns if col.endswith('A')]

    # Calculate the number of levels based on the dot count in 'RankID'
    filtered_data_combined['Hierarchy_Level'] = filtered_data_combined['RankID'].str.count(r'\.')

    # Add taxon names for later use
    taxon_mapping = dict(zip(sheet_data['RankID'], sheet_data['Taxon']))

    # Normalize values within each hierarchy level
    grouped = filtered_data_combined.groupby('Hierarchy_Level')
    aggregated_normalized = []

    for level, group in grouped:
        # Aggregate values within this hierarchy level
        level_aggregated = group.groupby('RankID')[[*before_columns, *after_columns]].sum()
        level_aggregated['Total_Before'] = level_aggregated[before_columns].sum(axis=1)
        level_aggregated['Total_After'] = level_aggregated[after_columns].sum(axis=1)
        
        # Normalize values within this level
        level_aggregated['Normalized_Before'] = level_aggregated['Total_Before'] / level_aggregated['Total_Before'].sum()
        level_aggregated['Normalized_After'] = level_aggregated['Total_After'] / level_aggregated['Total_After'].sum()
        
        # Add hierarchy level information
        level_aggregated['Hierarchy_Level'] = level
        
        # Append normalized results for this level
        aggregated_normalized.append(level_aggregated)

    # Combine results for all hierarchy levels
    aggregated_normalized_combined = pd.concat(aggregated_normalized)

    # Filter for significant contributions (>= threshold) within each hierarchy level
    filtered_hierarchy = aggregated_normalized_combined[
        (aggregated_normalized_combined['Normalized_Before'] >= threshold) |
        (aggregated_normalized_combined['Normalized_After'] >= threshold)
    ]

    # Add taxon names to the filtered data
    filtered_hierarchy['Taxon'] = filtered_hierarchy.index.map(taxon_mapping)

    # Reset index to make 'RankID' a column
    filtered_hierarchy = filtered_hierarchy.reset_index()

    # Add the Parent RankID
    def get_parent_rank_id(rank_id):
        if '.' in rank_id:
            return rank_id.rsplit('.', 1)[0]  # Remove the last segment after the last dot
        return None  # Top-level ranks do not have parents

    filtered_hierarchy['Parent_RankID'] = filtered_hierarchy['RankID'].map(get_parent_rank_id)

    # Add the Parent Taxon (optional)
    filtered_hierarchy['Parent_Taxon'] = filtered_hierarchy['Parent_RankID'].map(taxon_mapping)

    # Calculate Parent Rank Sizes (Total_Before for parents)
    parent_sizes = filtered_hierarchy.groupby('RankID')['Total_Before'].sum()
    filtered_hierarchy['Parent_Total_Before'] = filtered_hierarchy['Parent_RankID'].map(parent_sizes)

    # Reorder and restrict columns to specified ones
    filtered_hierarchy = filtered_hierarchy[
        ['Hierarchy_Level', 'RankID', 'Taxon', 'Parent_RankID', 'Parent_Taxon', 'Parent_Total_Before', 'Normalized_After', 'Normalized_Before']
    ]
    return filtered_hierarchy
 
def sort_hierarchy(data, start_level):
    """
    Sort hierarchical data based on 'Hierarchy_Level', propagating parent orders
    to child levels and sorting by 'Normalized_Before' within each parent group.
    """
    # Ensure levels are sorted for processing
    levels = sorted(data['Hierarchy_Level'].unique())
    
    if start_level not in levels:
        raise ValueError(f"Starting level {start_level} not found in data's hierarchy levels.")

    # Create an order dictionary to keep track of sorting at each level
    order_columns = {}
    sorted_data = None  # Initialize sorted_data to handle references before assignment

    # Iteratively process each level
    for level in levels:
        if level < start_level:
            continue  # Skip levels below the starting level
        
        if level == start_level:  # Start with the specified topmost level
            sorted_data = data[data['Hierarchy_Level'] == level].sort_values(
                by='Normalized_Before', ascending=False
            ).reset_index(drop=True)
            sorted_data[f'Level_{level}_Order'] = sorted_data.index + 1
        else:  # Process subsequent levels
            parent_level = level - 1
            # Ensure parent level is available in sorted_data
            if sorted_data is None or f'Level_{parent_level}_Order' not in sorted_data:
                raise ValueError(f"Parent level {parent_level} is missing required order data.")
            
            # Merge parent order into the current level
            data = data.merge(
                sorted_data[['RankID', f'Level_{parent_level}_Order']].rename(columns={'RankID': 'Parent_RankID'}),
                on='Parent_RankID',
                how='left'
            )
            # Sort current level based on parent order and 'Normalized_Before'
            sorted_data = data[data['Hierarchy_Level'] == level].sort_values(
                by=[f'Level_{parent_level}_Order', 'Normalized_Before'], ascending=[True, False]
            ).reset_index(drop=True)
            sorted_data[f'Level_{level}_Order'] = sorted_data.index + 1
        
        # Keep track of order columns for concatenation
        order_columns[level] = sorted_data

    # Combine sorted levels into a single DataFrame
    sorted_data = pd.concat(order_columns.values(), ignore_index=True)

    return sorted_data

def brighten_color(color, factor=0.5):
    """Brightens the given RGB color by mixing it with white."""
    color = mcolors.to_rgb(color)  # Ensure color is in RGB
    return tuple(1 - factor * (1 - np.array(color)))  # Blend with white

def plot_taxonomy_chart(filtered_hierarchy, output_svg_path, start_rank, ann_threshold):
    """
    Plot semi-pie charts for N hierarchical levels.
    :param filtered_hierarchy: DataFrame with hierarchical data.
    :param output_svg_path: Path to save the output chart (supports .svg or .png).
    :param start_rank: The starting hierarchy rank.
    :param ann_threshold: Threshold for annotation visibility.
    """
    # Determine the unique hierarchy levels
    unique_levels = sorted(filtered_hierarchy['Hierarchy_Level'].unique())
    num_levels = len(unique_levels)
    
    # Define radius range for each level dynamically
    radius_step = 0.3
    base_radius = 0.7

    # Initialize the figure and axis
    fig, ax = plt.subplots(figsize=(12, 12))

    # Initialize parent color mapping
    parent_color_mapping = {}
    skip_color_mapping = False

    for level_idx, hierarchy_level in enumerate(unique_levels):
        # Filter data for the current hierarchy level
        level_data = filtered_hierarchy[filtered_hierarchy['Hierarchy_Level'] == hierarchy_level]
        
        # Check if this level has only one taxonomy
        unique_taxonomies = level_data['Taxon'].nunique()
        if hierarchy_level == start_rank and unique_taxonomies == 1:
            # Mark this level to skip its color mapping but keep it in the chart
            skip_color_mapping = True
        elif unique_taxonomies > 1:
            # Reset the flag if subsequent levels have more than one taxonomy
            skip_color_mapping = False

        # Assign colors for the current level
        if not parent_color_mapping or (skip_color_mapping and hierarchy_level == start_rank):
            # Base colors for the first valid level or skip coloring
            level_colors = plt.cm.Paired(np.linspace(0, 1, len(level_data)))
            if not skip_color_mapping:
                parent_color_mapping = dict(zip(level_data['RankID'], level_colors))
        else:
            # Map colors based on parent, starting from adjusted level
            level_colors = [
                brighten_color(parent_color_mapping.get(parent_rank_id, (0.5, 0.5, 0.5)), factor=0.7)
                for parent_rank_id in level_data['Parent_RankID']
            ]
            # Update the color mapping for the current level
            parent_color_mapping.update(dict(zip(level_data['RankID'], level_colors)))
        
        # Normalize "Before" and "After" values
        sizes_before = level_data['Normalized_Before'].values
        sizes_after = level_data['Normalized_After'].values
        labels = level_data['Taxon'].values

        # Define radii for this level
        outer_radius = base_radius + radius_step * (level_idx - unique_levels.index(start_rank))
        inner_radius = outer_radius - 0.3

        # Top half (After)
        sizes_top = np.concatenate([sizes_after, [sum(sizes_after)]])  # Add blank
        colors_top = list(level_colors) + [(0, 0, 0, 0)]  # Transparent for blank
        wedges_top, _ = ax.pie(
            sizes_top,
            radius=outer_radius,
            startangle=0,
            colors=colors_top,
            wedgeprops=dict(width=0.3, edgecolor='k', linewidth=0.5),
            labeldistance=1.05
        )

        # Bottom half (Before)
        sizes_bottom = np.concatenate([sizes_before, [sum(sizes_before)]])  # Add blank
        colors_bottom = list(level_colors) + [(0, 0, 0, 0)]  # Transparent for blank
        wedges_bottom, _ = ax.pie(
            sizes_bottom,
            radius=outer_radius,
            startangle=180,
            colors=colors_bottom,
            wedgeprops=dict(width=0.3, edgecolor='k', linewidth=0.5),
            labeldistance=1.05
        )

        # Annotate the wedges for the current level
        for i, wedge in enumerate(wedges_top[:-1]):  # Top (After) annotations
            if sizes_after[i] >= ann_threshold:
                angle = (wedge.theta2 + wedge.theta1) / 2
                x = (inner_radius + 0.15) * np.cos(np.radians(angle))
                y = (inner_radius + 0.15) * np.sin(np.radians(angle))
                ax.text(
                    x, y, f'{labels[i]}\n{sizes_after[i]*100:.1f}%', 
                    ha='center', va='center', fontsize=8
                )
        
        for i, wedge in enumerate(wedges_bottom[:-1]):  # Bottom (Before) annotations
            if sizes_before[i] >= ann_threshold:
                angle = (wedge.theta2 + wedge.theta1) / 2
                x = (inner_radius + 0.15) * np.cos(np.radians(angle))
                y = (inner_radius + 0.15) * np.sin(np.radians(angle))  # Flip for bottom
                ax.text(
                    x, y, f'{labels[i]}\n{sizes_before[i]*100:.1f}%', 
                    ha='center', va='center', fontsize=8
                )

    # Calculate the diameter of the largest circle
    max_radius = base_radius + radius_step * (len(unique_levels) - 1)

    # Add a dividing line for top and bottom halves
    ax.plot([-max_radius, max_radius], [0, 0], color='black', linewidth=1.5)

    # Set plot limits to ensure the dividing line fits
    ax.set_xlim(-max_radius, max_radius)
    ax.set_ylim(-max_radius, max_radius)

    # Add "Before" and "After" labels in the center
    ax.text(0, -0.15, "Before", ha='center', va='center', fontsize=12, color='black', fontweight='normal')
    ax.text(0, 0.15, "After", ha='center', va='center', fontsize=12, color='black', fontweight='normal')

    # Determine output format and save the plot
    _, file_extension = os.path.splitext(output_svg_path)
    if file_extension.lower() == '.svg':
        fig.savefig(output_svg_path, format='svg', bbox_inches='tight')
        print(f"Chart exported successfully as SVG: {output_svg_path}")
    elif file_extension.lower() == '.png':
        fig.savefig(output_svg_path, format='png', bbox_inches='tight', dpi=300)
        print(f"Chart exported successfully as PNG: {output_svg_path}")
    else:
        print("Invalid file format. Please specify a .svg or .png file path.")

    # Show the plot
    plt.show()

# Load the Excel file
file_path = 'C:/Users/a/Desktop/Main.xlsx'  # Replace with your actual file path
output_svg_path = 'output.png' # Replace with your path to save the chart, .png for PNG format, .svg for the SVG format
output_path = 'output.xlsx' # Replace with your path to save edited xlsx file
sheet_data = pd.read_excel(file_path, sheet_name='Sheet1') # Import the sheet

# Define starting rank and total taxonomy levels to process
start_rank = 1  # Starting rank for filtering
taxonomy = 4 # Number of taxonomy levels to process
threshold = 0.001 # The threshold to limit the taxonomies shown on the graph (%) 
ann_threshold =0.05 # The threshold to limit the annotation's of the taxonomies shown on the graph (%) 

# Function to process the raw dataset
filtered_hierarchy=process_xlsx(sheet_data,start_rank,taxonomy,threshold)

# Function to sort the taxonomies hierarchic and percentage order
filtered_hierarchy=sort_hierarchy(filtered_hierarchy,start_rank)

filtered_hierarchy.to_excel(output_path, index=False) # Export to Excel

# Function to draw the chart
plot_taxonomy_chart(filtered_hierarchy, output_svg_path ,start_rank, ann_threshold)