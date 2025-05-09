import optuna
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Optuna study configuration
DB_PATH = "sqlite:///hp_opt/VideoLLaVA/optuna_newborn.db"  
STUDY_NAME = "newborn_activity_recognition"

def main():
    # Load the study
    study = optuna.load_study(study_name=STUDY_NAME, storage=DB_PATH)

    # Get all completed trials
    completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]

    if not completed_trials:
        logger.info("No completed trials found.")
        return

    # Sort by objective value (descending for maximization)
    sorted_trials = sorted(completed_trials, key=lambda x: x.value, reverse=True)

    # Display top 5 trials
    top_n = 5
    logger.info(f"Top {top_n} trials so far:")
    for i, trial in enumerate(sorted_trials[:top_n]):
        logger.info(f"\nRank {i+1}:")
        logger.info(f"  Trial #{trial.number} - F1 Score: {trial.value:.4f}")
        for key, val in trial.params.items():
            logger.info(f"  {key}: {val}")

    # Display the best trial so far
    best_trial = study.best_trial
    logger.info("\nBest Trial So Far:")
    logger.info(f"  Trial #{best_trial.number} - F1 Score: {best_trial.value:.4f}")
    for key, val in best_trial.params.items():
        logger.info(f"  {key}: {val}")

if __name__ == "__main__":
    main()
