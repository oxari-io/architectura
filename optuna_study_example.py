import optuna
import sqlite3

# Define the objective function to optimize
def objective(trial):
    x = trial.suggest_uniform('x', -10, 10)
    y = trial.suggest_uniform('y', -10, 10)
    return (x - 2) ** 2 + (y + 3) ** 2

# Specify the SQLite database file for Optuna study
study_name = 'optuna_study_example'
storage_name = 'sqlite:///optuna_study_example.db'

# Create an Optuna study and store it in the SQLite database
study = optuna.create_study(study_name=study_name, storage=storage_name, direction='minimize')

# Optimize the study (replace this with your actual optimization task)
study.optimize(objective, n_trials=100)

# Connect to the SQLite database and inspect the results
connection = sqlite3.connect('optuna_study_example.db')
cursor = connection.cursor()

# Print the best trial parameters and result
best_trial = study.best_trial
print(f"Best trial - Trial number: {best_trial.number}, Value: {best_trial.value}")
print(f"Params: {best_trial.params}")

# Close the database connection
connection.close()
