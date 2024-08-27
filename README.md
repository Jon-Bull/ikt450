# ikt450

## Table of Contents

1. [Preparations](#preparations)
   - [Clone the Repository](#clone-the-repository)
   - [Setting Up a Virtual Environment](#setting-up-a-virtual-environment)
     - [Using `venv`](#using-venv-pythons-built-in-tool)
     - [Using `conda`](#using-conda-if-you-prefer-using-anacondaminiconda)
   - [Setting Up `PYTHONPATH`](#setting-up-pythonpath)
     - [On Linux or macOS](#on-linux-or-macos)
     - [On Windows](#on-windows)
2. [Usage](#usage)
   - [For each feature 🔃](#for-each-feature-🔃)


## Preparations
### Clone the Repository:
```bash
git clone git@github.com:Jon-Bull/ikt450.git
cd ikt450/
```

### Setting Up a Virtual Environment

<details>
<summary><i>Show steps</i></summary>
<br>


#### Using `venv` (Python's built-in tool):
<details>
<summary><i>Show steps</i></summary>
<br>

1. **Create a Virtual Environment**:
   ```bash
   python3 -m venv env
   ```
   This will create a directory named `env` that contains your virtual environment.

2. **Activate the Virtual Environment**:
   - On Linux or macOS:
     ```bash
     source env/bin/activate
     ```
   - On Windows:
     ```cmd
     .\env\Scripts\activate
     ```

3. **Install Required Packages**:
   ```bash
   pip install -r requirements.txt
   ```
</details>

#### Using `conda` (if you prefer using Anaconda/Miniconda):
<details>
<summary><i>Show steps</i></summary>
<br>

1. **Create a Conda Environment**:
   ```bash
   conda create --name ikt450 python=3.x
   ```
   Replace `3.x` with the desired Python version.

2. **Activate the Conda Environment**:
   ```bash
   conda activate ikt450
   ```

3. **Install Required Packages**:
   ```bash
   pip install -r requirements.txt
   ```

</details>
</details>

### Setting Up `PYTHONPATH`
<details>
<summary><i>Show steps</i></summary>
<br>

To ensure that Python can properly locate the modules within this project, you should add the project's root directory to your `PYTHONPATH` environment variable. This can be done by adding the following line to your shell configuration file (e.g., `.bashrc`, `.zshrc`, `.bash_profile`, etc.), depending on your operating system:

#### On Linux or macOS:
<details open>
<summary><i>Show steps</i></summary>
<br>

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

</details>

#### On Windows:
<details open>
<summary><i>Show steps</i></summary>
<br>

1. Open Command Prompt as an administrator.
2. Set the `PYTHONPATH` for your session:
   ```cmd
   set PYTHONPATH=%PYTHONPATH%;C:\REPLACE\THIS\PATH\ikt450\
   ```
3. To set it permanently, use the `setx` command:
   ```cmd
   setx PYTHONPATH "%PYTHONPATH%;C:\REPLACE\THIS\PATH\ikt450\"
   ```

</details>
</details>

## Usage 

<details>
<summary><i>Show steps</i></summary>
<br>

### For each feature 🔃 

|              Before              |              Coding...            |              After               |
:---------------------------------:|:---------------------------------:|:---------------------------------:
| ```git pull``` <br /> ```git checkout -b <feature>```  |     🌟 WRITE AWESOME CODE 🌟      | ```git add .``` <br /> ```git commit -m "<message>"``` <br /> ```git push -u origin <feature>```|

▶️ Open a pull request on [GitHub](https://github.com/Jon-Bull/ikt450/pulls) to merge your changes into the main branch.

▶️ Review the changes in the pull request and make any necessary comments or suggestions.

▶️ Once the changes are approved, merge the pull request into the main branch.

▶️ Switch back to the main branch and pull the latest changes:
	
	git checkout main
	git pull
	
</details>