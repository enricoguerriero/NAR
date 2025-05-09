import optuna
import time
import datetime
import os

def estimate_remaining_time(study_name, db_path):
    """
    Estimates remaining time for ongoing Optuna study based on elapsed time per trial.

    Args:
    - study_name (str): Name of the Optuna study.
    - db_path (str): Path to the SQLite database.

    Returns:
    - None: Prints the estimated remaining time.
    """
    # Load the study
    study = optuna.load_study(study_name=study_name, storage=db_path)
    completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]

    if not completed_trials:
        print("No completed trials yet. Cannot estimate remaining time.")
        return

    # Calculate elapsed time
    start_time = completed_trials[0].datetime_start.timestamp()
    current_time = time.time()
    elapsed_time = current_time - start_time

    # Calculate average time per trial
    avg_time_per_trial = elapsed_time / len(completed_trials)

    # Total number of trials set initially
    total_trials = max(t.number for t in study.trials) + 1

    # Remaining trials
    remaining_trials = total_trials - len(completed_trials)

    # Estimated remaining time
    estimated_remaining_time = avg_time_per_trial * remaining_trials

    # Output
    print(f"Completed Trials: {len(completed_trials)} / {total_trials}")
    print(f"Elapsed Time: {datetime.timedelta(seconds=elapsed_time)}")
    print(f"Estimated Remaining Time: {datetime.timedelta(seconds=estimated_remaining_time)}")
    print(f"Expected Completion Time: {datetime.datetime.now() + datetime.timedelta(seconds=estimated_remaining_time)}")


# Example usage:
if __name__ == "__main__":
    DB_PATH = "sqlite:///hp_opt/VideoLLaVA/optuna_newborn.db"  # Adjust path if needed
    STUDY_NAME = "newborn_activity_recognition"
    
    estimate_remaining_time(STUDY_NAME, DB_PATH)
