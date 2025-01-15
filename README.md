# 🩺 Breast Cancer Detection Project 🩺

## Contributors
1. **David Asamoa Berko**
2. **Ekow Essel Hazel Davis**
3. **Ewurafua Quansah**

This project aims to develop a machine learning model to detect breast cancer using logistic regression. The project includes data cleaning, model training, and evaluation.

## Project Structure

```
- 📁 .gitignore
- 📁 .idea/
- 📁 app/
- 📁 data/
- 🐍 [main.py](main.py)
- 📁 models/
- 📁 myenv/
- 📁 notebooks/
- 📄 [README.md](README.md)
- 📁 utils/
```

### 📂 Directories

- **app/**: Contains the main application code.
    - `app.py`: Main application script.
- **data/**: Contains the dataset used for training and evaluation.
    - `BC_data.csv`: Breast cancer dataset.
- **models/**: Contains the machine learning models and related scripts.
    - `logistic_model.pkl`: Trained logistic regression model.
    - `logistic_regression.py`: Script for training the logistic regression model.
- **myenv/**: Virtual environment for the project.
- **notebooks/**: Jupyter notebooks for data cleaning and model documentation.
    - `Breast Cancer Detection Data Cleaning Documentation.ipynb`
    - `Breast Cancer Detection Model Documentation.ipynb`
- **utils/**: Utility scripts and functions.

### 📄 Files

- **main.py**: Entry point for the project.
- **README.md**: Project documentation (this file).

## 🚀 Setup

1. Clone the repository.
2. Create a virtual environment and activate it:
     ```sh
     python -m venv myenv
     source myenv/bin/activate  # On Windows use `myenv\Scripts\activate`
     ```
3. Install the required packages:
     ```bash
     pip install -r requirements.txt
     ```

## 🛠️ Usage

1. Run the main application:
     ```bash
     python main.py
     ```
2. Explore the Jupyter notebooks in the [notebooks](notebooks/) directory for data cleaning and model documentation.

## 📜 License

This project is licensed under the 0-clause BSD License. See the [LICENSE](LICENSE.txt) file for more details.