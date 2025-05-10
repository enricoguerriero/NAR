import optuna
import logging
from argparse import ArgumentParser
from optuna.visualization import (
    plot_optimization_history,
    plot_param_importances,
    plot_parallel_coordinate,
    plot_slice,
    plot_contour
)
import matplotlib.pyplot as plt
import os

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Optuna study configuration
def main():
    
    parser = ArgumentParser(description="Monitor Optuna Study")
    parser.add_argument("--model_name", type=str, required=True, help="Name of the model to monitor.")
    args = parser.parse_args()
    model_name = args.model_name
    logger.info(f"Monitoring Optuna study for model: {model_name}")
    
    DB_PATH = f"sqlite:///hp_opt/{model_name}/optuna_newborn.db"  
    STUDY_NAME = "newborn_activity_recognition"
    
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

    # Generate and save plots
    figs = {
        'optimization_history': plot_optimization_history(study),
        'param_importances': plot_param_importances(study),
        'parallel_coordinate': plot_parallel_coordinate(study),
        'slice_plot': plot_slice(study),
        'contour_plot': plot_contour(study, params=['learning_rate', 'weight_decay'])
    }

    output_dir = f"plots/{model_name}/"
    os.makedirs(output_dir, exist_ok=True)
    
    for name, fig in figs.items():
        fig_path = f"{output_dir}/{name}.png"
        fig.savefig(fig_path)
        logger.info(f"Saved {name} plot at: {fig_path}")

    logger.info("All plots saved successfully.")

if __name__ == "__main__":
    main()
