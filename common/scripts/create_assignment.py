import os
import shutil

def create_assignment_folders(base_path, assignment_name, template_path):
    assignment_dir = os.path.join(base_path, assignment_name)
    
    # Define the structure
    structure = {
        "data": {},
        "drafts": {},
        "models": {},
        "notebooks": {},
        "scripts": {},
        "results": {},
    }
    
    # Create the base assignment directory
    os.makedirs(assignment_dir, exist_ok=True)
    
    # Create subdirectories and add placeholder README files
    for folder, subfolders in structure.items():
        folder_path = os.path.join(assignment_dir, folder)
        os.makedirs(folder_path, exist_ok=True)
        
        readme_path = os.path.join(folder_path, "README.md")
        with open(readme_path, "w") as f:
            f.write(f"# {folder.capitalize()} for {assignment_name}\n\n")
            f.write(f"This directory contains files related to {folder}.\n")
    
    # Copy the notebook template into the notebooks directory
    notebook_template_dest = os.path.join(assignment_dir, "notebooks", "ml_notebook_template.ipynb")
    shutil.copy(template_path, notebook_template_dest)
    
    # Create a main README and a report file
    with open(os.path.join(assignment_dir, "README.md"), "w") as f:
        f.write(f"# {assignment_name}\n\n")
        f.write(f"Details about the assignment and instructions on how to run the code.\n")
    
    with open(os.path.join(assignment_dir, "report.md"), "w") as f:
        f.write(f"# {assignment_name} Report\n\n")
        f.write(f"Summary of the findings, methodologies, and results for the {assignment_name} assignment.\n")
    
    print(f"Assignment '{assignment_name}' has been created successfully at '{assignment_dir}'.")

def main():
    # Determine the base directory relative to the location of the script
    script_dir = os.path.dirname(os.path.realpath(__file__))
    default_base_path = os.path.join(script_dir, "../../assignments/")
    default_base_path = os.path.abspath(default_base_path)  # Resolve to an absolute path
    
    # Prompt the user with the pre-filled base path, allowing edits
    base_path = input(f"Enter the base path for assignments directory. Press ENTER if this is correct [{default_base_path}]: ").strip()
    
    # Use the default path if the user does not provide any input
    if not base_path:
        base_path = default_base_path
    
    # Ask for the assignment name
    assignment_name = input("Enter the name of the new assignment (e.g., 2_assignment_name): ").strip()
    
    # Path to the notebook template
    template_path = os.path.join(script_dir, "../notebooks/ml_notebook_template.ipynb")
    template_path = os.path.abspath(template_path)  # Ensure the path is absolute
    
    # Create the folder structure
    create_assignment_folders(base_path, assignment_name, template_path)

if __name__ == "__main__":
    main()
