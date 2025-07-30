# Employee Salary Prediction

This project is a web application that predicts employee salaries based on experience, education, job role, location, and industry. It uses a machine learning model trained on synthetic data and provides a user-friendly interface for predictions.

## Features

- Predicts employee salary based on user input.
- Trained using both Linear Regression and Random Forest; the best model is automatically selected.
- Web interface built with Flask and Bootstrap.
- Synthetic dataset generation script included.

## Project Structure

- [`app.py`](app.py): Flask web application for salary prediction.
- [`train_and_save_model.py`](train_and_save_model.py): Trains machine learning models and saves the best one.
- [`generate_dataset.py`](generate_dataset.py): Generates a synthetic dataset for training.
- [`employee_salary_data.csv`](employee_salary_data.csv): The dataset used for training.
- [`salary_predictor.joblib`](salary_predictor.joblib): The trained model file.
- [`requirements.txt`](requirements.txt): Python dependencies.
- [`templates/index.html`](templates/index.html): HTML template for the web interface.

## Setup Instructions

1. **Clone the repository** and navigate to the project directory.

2. **Install dependencies**:
    ```sh
    pip install -r requirements.txt
    ```

3. **Generate the dataset** (optional, if you want to regenerate):
    ```sh
    python generate_dataset.py
    ```
    > Note: The generated file is saved as `employee_salary_data.csv`.

4. **Train the model**:
    ```sh
    python train_and_save_model.py
    ```
    > This will create or update `salary_predictor.joblib`.

5. **Run the web application**:
    ```sh
    python app.py
    ```
    The app will be available at [http://127.0.0.1:5000/](http://127.0.0.1:5000/).

## Usage

- Open the web app in your browser.
- Enter the required details (experience, education, job role, location, industry).
- Click "Predict Salary" to see the predicted salary.

## Notes

- The dataset and model are synthetic and for demonstration purposes.
- The dropdown options in the web app are currently set for Indian cities, but the dataset uses Indian cities. You may want to align these for consistency.

## License

This project is for educational
