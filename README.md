# Gut Microbiome Taxonomy Data Visualization Tool

In a multidisciplinary brain-gut axis study we needed a tool to demonstrate the gut microbiome result in Krona style. Since Krona is not working in Microsoft properly, we developed this tool. This Python-based tool provides a comprehensive solution for processing, sorting, and visualizing hierarchical taxonomy data from an Excel file. It is designed to enable insightful visualization in the form of hierarchical pie charts, for conditional studies.

## Description

The tool processes taxonomy data from an Excel file, computes normalized values at each hierarchy level, filters data based on specified thresholds, and generates a semi-pie chart to visualize the "Before" and "After" distributions of each taxon. Additionally, the processed data can be exported to a new Excel file for further analysis.

## Features

1. **Data Processing**:
   - Normalizes data at each hierarchy level.
   - Filters taxonomies based on significance thresholds.
   - Computes parent-child relationships between taxonomies.

2. **Sorting**:
   - Sorts taxonomies hierarchically and by percentage values.
   - Ensures logical ordering across levels.

3. **Visualization**:
   - Generates hierarchical semi-pie charts.
   - Displays "Before" and "After" distributions for each level.
   - Annotates significant taxonomies based on user-defined thresholds.

4. **Export**:
   - Outputs processed and sorted taxonomy data to an Excel file.

## Prerequisites

To use this tool, ensure you have the following installed:

- **Python 3.8+**: The script is written in Python and requires a compatible environment.
- **Python Libraries**:
  - `pandas`: For data processing.
  - `numpy`: For numerical operations.
  - `matplotlib`: For visualization.

- Install the required libraries via `pip`:
  - `pip install pandas numpy matplotlib`

## Usage

### Prepare Input Data:
- Place your Excel file (`Main.xlsx`) in the desired directory.
- Ensure the file contains a `Sheet1` with columns like `RankID`, `Taxon`, and other numeric data.

### Update File Paths:
- Update the `file_path` variable in the script with the path to your Excel file.
- Set the `output_svg_path` for saving the visualization.
- Specify the `output_path` for exporting the processed Excel file.

### Customize Parameters:
- Adjust `start_rank` to define the starting hierarchy level.
- Set `taxonomy` to the number of levels to process.
- Modify `threshold` and `ann_threshold` for filtering and annotation visibility.

### Run the Script:
- Execute the script in a Python environment.

### Output:
- The processed data will be saved to the specified Excel file (`output_path`).
- The hierarchical chart will be saved as an image (`output_svg_path`).

## Process

### Data Processing
- The script reads the Excel file and filters ranks based on hierarchy levels and a threshold.
- It computes normalized values for "Before" and "After" distributions.

### Hierarchy Sorting
- The taxonomies are sorted hierarchically and by normalized percentages.

### Visualization
- A semi-pie chart is created, showing the "Before" and "After" contributions at each hierarchy level.

### Export
- Processed data is saved to a new Excel file.

## Notes
- Ensure the Excel file follows the required format with appropriate column names.
- For large datasets, adjust the parameters to optimize performance and chart clarity.

## Example

### Input Excel File

In our study, the analysis result for the gut microbiome was given in this format: The columns ending with B show the before values, and the ones with A show the after values.  

| RankID  | Taxon              | Value1B | Value1A | Value2B | Value2A |
|---------|--------------------|---------|---------|---------|---------|
| 0.1     | Bacteria           | 1000    | 1500    | 1200    | 1800    |
| 0.1.1   | Actinomycetota     | 500     | 700     | 600     | 800     |
| 0.1.1.1 | Actinomycetes      | 200     | 300     | 250     | 350     |
| 0.1.2   | Bacillota          | 300     | 400     | 350     | 450     |
| 0.1.3   | Bacteroidota       | 150     | 250     | 180     | 280     |

### Processed Output

| Hierarchy_Level | RankID | Taxon   | Parent_RankID | Parent_Taxon | Parent_Total_Before | Normalized_Before | Normalized_After |
|-----------------|--------|---------|---------------|--------------|---------------------|-------------------|------------------|
| 1               | 1      | Kingdom | None          | None         | N/A                 | 1.0               | 1.0              |
| 2               | 1.1    | Phylum  | 1             | Kingdom      | 100                 | 0.5               | 0.5              |
| 3               | 1.1.1  | Class   | 1.1           | Phylum       | 50                  | 0.4               | 0.4              |

### Visualization
The tool generates a hierarchical semi-pie chart displaying the normalized contributions of each taxon.

### Output Example
![Example](https://github.com/user-attachments/assets/d52bf663-5d2d-4d57-8978-3cda70f1f88e)

## License
Feel free to use, modify, and distribute it.

## Contributions
Contributions are welcome! If you encounter any issues or have suggestions for improvements, please create an issue or a pull request in the repository.
