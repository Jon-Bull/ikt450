# Assignment Folder Creation Script

This script is designed to automate the creation of a folder structure for new assignments in a course or project. It sets up the necessary directories, adds placeholder `README.md` files, and copies a notebook template to the appropriate directory.

## How to Use

### Prerequisites

- Ensure that Python is installed on your system.
- The script should be placed in a directory where it can locate the necessary notebook template file (`ml_notebook_template.ipynb`).

### Steps

1. **Navigate to the Script Directory**:
   - Open a terminal and navigate to the directory where the script is located.

2. **Run the Script**:
   - Execute the script using the following command:
     ```bash
     python3 create_assignment.py
     ```
   - If your system uses Python 3 as the default Python version, you can also run it using:
     ```bash
     python create_assignment.py
     ```

3. **Input the Required Information**:
   - The script will prompt you for the base path where the assignments should be created. A default path will be provided based on the script's location. Press `ENTER` to accept the default or provide a custom path.
   - Enter the name of the new assignment (e.g., `2_assignment_name`). This will be used to name the assignment folder.

4. **Folder Structure Created**:
   - The script will create the following folder structure under the specified assignment name:
     ```
     ├── data/
     ├── drafts/
     ├── models/
     ├── notebooks/
     ├── scripts/
     ├── results/
     ├── README.md
     └── report.md
     ```
   - Each folder will contain a `README.md` file describing its purpose.
   - The notebook template (`ml_notebook_template.ipynb`) will be copied to the `notebooks/` directory.

5. **Verify the Output**:
   - After running the script, navigate to the newly created assignment directory to verify that the folders and files have been correctly set up.

### Example

```bash
$ python3 create_assignment.py
Enter the base path for assignments directory. Press ENTER if this is correct [/path/to/your/projects/assignments/]: 
Enter the name of the new assignment (e.g., 2_assignment_name): 3_logistic_regression
Assignment '3_logistic_regression' has been created successfully at '/path/to/your/projects/assignments/3_logistic_regression'
