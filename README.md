# ikt450


# Preparations
### Clone repo:
	git clone git@github.com:Jon-Bull/ikt450.git
    cd ikt450/

### Setting Up `PYTHONPATH`

To ensure that Python can properly locate the modules within this project, you should add the project's root directory to your `PYTHONPATH` environment variable. This can be done by adding the following line to your shell configuration file (e.g., `.bashrc`, `.zshrc`, `.bash_profile`, etc.), depending on your operating system:

#### On Linux or macOS:
1. Open your terminal.
2. Edit your `.bashrc` or `.zshrc` file:
   ```bash
   nano ~/.bashrc
   ```
   or for zsh:
   ```bash
   nano ~/.zshrc
   ```
3. Add the following line at the end of the file:
   ```bash
   export PYTHONPATH="${PYTHONPATH}:/REPLACE/THIS/PATH/ikt450/"
   ```
4. Save and close the file.
5. Apply the changes by sourcing the file:
   ```bash
   source ~/.bashrc
   ```
   or for zsh:
   ```bash
   source ~/.zshrc
   ```

#### On Windows:
1. Open Command Prompt as an administrator.
2. Set the `PYTHONPATH` for your session:
   ```cmd
   set PYTHONPATH=%PYTHONPATH%;C:\REPLACE\THIS\PATH\ikt450\
   ```
3. To set it permanently, use the `setx` command:
   ```cmd
   setx PYTHONPATH "%PYTHONPATH%;C:\REPLACE\THIS\PATH\ikt450\"
   ```

# Usage 

## For each feature 🔃 

|              Before              |              Coding...            |              After               |
:---------------------------------:|:---------------------------------:|:---------------------------------:
| ```git pull``` <br /> ```git checkout -b <feature>```  |     🌟 WRITE AWESOME CODE 🌟      | ```git add .``` <br /> ```git commit -m "<message>"``` <br /> ```git push -u origin <feature>```|

▶️ Open a pull request on [GitHub](https://github.com/Jon-Bull/ikt450/pulls) to merge your changes into the main branch.

▶️ Review the changes in the pull request and make any necessary comments or suggestions.

▶️ Once the changes are approved, merge the pull request into the main branch.

▶️ Switch back to the main branch and pull the latest changes:
	
	git checkout main
	git pull
	
