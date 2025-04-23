# Course Project – K-Means
Full project outline can be read in course-project.pdf

## How to Run the Code
### Prerequisites
1. **Python**: Ensure python is installed on your system.
2. **Libraries**: Install the following Python libraries:
```bash
pip install numpy pandas matplotlib scikit-learn scipy
```
3. **Datasets**: Ensure the following datasets are in the same directory as `kmeans.py`:
- ```CA-GrQc.txt```
- ```com-dblp.ungraph.txt```
   - **Note for Mac Users**: If the large dataset (```com-dblp.ungraph.txt```) has been moved to iCloud, even if in the same directory, you must redownload it to your local machine. The script cannot access files stored in the cloud.
### Running the Code
1. Place the datasets (```CA-GrQc.txt``` and ```com-dblp.ungraph.txt```) in the same directory as ```kmeans.py```.
2. Run the script using Python:
```bash
python kmeans.py
```
3. Follow the prompts:
     - **First Dataset (Directed)**:
        - Enter the number of clusters (k) when prompted.
        - The script will display the effectiveness scores (Silhouette Coefficient and Davies-Bouldin Index) and runtime.
        - A pop-up window will display a visualization of the clustering. Close the window to proceed.
     - **Second Dataset (Undirected)**:
        - Enter the number of clusters (k) when prompted.
        - The script will calculate the effectiveness scores and runtime. Visualization is skipped for this dataset due to its size.
### Notes
- **Runtime for Large Dataset (```com-dblp.ungraph.txt```)**:
    - On a 2020 Mac with an M1 chip, the large dataset (com-dblp.ungraph.txt) took on average 940 seconds (15-18 minutes) to process.
- **Visualization**:
    - The visualization for the smaller dataset will appear as a pop-up window. Ensure you close the window to continue the script.
    - Visualization for the larger dataset is skipped to save time.
## Authors
- Demarco Guajardo
- Richard Harris
