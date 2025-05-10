import optuna
import torch
from torch.utils.data import DataLoader
from models.timesformer import TimeSformer  
from data.token_dataset import TokenDataset
import logging
from optuna.integration.wandb import WeightsAndBiasesCallback
from optuna.visualization import (
    plot_optimization_history,
    plot_param_importances,
    plot_parallel_coordinate,
    plot_slice,
    plot_contour
)
import wandb

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PROJECT_NAME = "NewbornActivityRecognition"
DB_PATH = "sqlite:///optuna_newborn.db"  # Path to persistent storage
STUDY_NAME = "newborn_activity_recognition"


def objective(trial: optuna.Trial) -> float:
    """
    Objective function for Optuna to optimize.
    Returns the validation macro F1 score.
    """
    logger.info(f"Trial number: {trial.number}")
    wandb.init(
        project=PROJECT_NAME,
        reinit=True,
        config={},
        name=f"TS_optim_trial_{trial.number}"
    )

    # Hyperparameter sampling
    optimizer_name = trial.suggest_categorical('optimizer_name', ['adam', 'sgd', 'adamw'])
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-2)
    weight_decay = trial.suggest_loguniform('weight_decay', 1e-6, 1e-2)
    epochs = trial.suggest_int('epochs', 10, 50, step=5)
    momentum = None
    if optimizer_name in ('sgd', 'adamw'):
        momentum = trial.suggest_uniform('momentum', 0.0, 0.99)

    criterion_name = "wbce"
    pos_weight = torch.tensor([0.19311390817165375, 2.532083511352539,
                               7.530612468719482, 6.510387420654297]).to(device)
    prior_probability = torch.tensor([0.8381429314613342, 0.2831190228462219,
                                      0.11722487956285477, 0.1331489235162735]).to(device)
    threshold = trial.suggest_float('threshold', 0.3, 0.7)

    scheduler_name = trial.suggest_categorical(
        'scheduler_name', ['steplr', 'cosineannealinglr', 'reduceonplateau']
    )
    scheduler_patience = trial.suggest_int('scheduler_patience', 1, 5)
    patience = trial.suggest_int('patience', 3, 8)
    freezing_condition = trial.suggest_categorical('freezing_condition', ['none', 'all', 'partial'])

    # Log config to W&B
    wandb.config.update({
        'optimizer_name': optimizer_name,
        'learning_rate': learning_rate,
        'epochs': epochs,
        'weight_decay': weight_decay,
        'momentum': momentum,
        'criterion_name': criterion_name,
        'threshold': threshold,
        'scheduler_name': scheduler_name,
        'scheduler_patience': scheduler_patience,
        'patience': patience,
        'freezing_condition': freezing_condition
    })

    # Model initialization
    model = TimeSformer(device=device).to(device)
    
    # Training
    results = model.train_from_tokens(
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        epochs=epochs,
        optimizer_name=optimizer_name,
        learning_rate=learning_rate,
        momentum=momentum,
        weight_decay=weight_decay,
        criterion_name=criterion_name,
        pos_weight=pos_weight,
        threshold=threshold,
        scheduler_name=scheduler_name,
        scheduler_patience=scheduler_patience,
        patience=patience,
        show_progress=False,
        prior_probability=prior_probability,
        logger=logger
    )

    # Log and return metric
    val_f1 = results['val_metrics']['f1_macro']
    wandb.log({'val_f1_macro': val_f1})
    wandb.finish()
    return val_f1


def collate_fn(batch):
    """
    Collate function for DataLoader.
    """
    pixel_values = torch.cat([item["pixel_values"] for item in batch], dim=0)
    labels = torch.stack([item["labels"] for item in batch], dim=0)
    return {"pixel_values": pixel_values, "labels": labels}


if __name__ == '__main__':
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    logger.info("Starting Optuna study with persistence.")

    # Data loading
    train_dataset = TokenDataset('data/tokens/TimeSformer/train/2sec_4fps')
    val_dataset = TokenDataset('data/tokens/TimeSformer/validation/2sec_4fps')
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True,
                              num_workers=4, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False,
                            num_workers=4, collate_fn=collate_fn)

    # Optuna study: create or load existing
    study = optuna.create_study(
        study_name=STUDY_NAME,
        storage=DB_PATH,
        direction='maximize',
        sampler=optuna.samplers.TPESampler(),
        pruner=optuna.pruners.MedianPruner(),
        load_if_exists=True
    )

    # WandB integration
    wandb_callback = WeightsAndBiasesCallback(
        metric_name="val_f1_macro",
        wandb_kwargs={"project": PROJECT_NAME, "reinit": True}
    )

    # Optimize (will resume missing trials automatically)
    study.optimize(objective, n_trials=100, n_jobs=1, callbacks=[wandb_callback])

    # Log best trial
    best = study.best_trial
    logger.info(f"Best trial: #{best.number} Value: {best.value:.4f}")
    logger.info("Params:")
    for key, val in best.params.items():
        logger.info(f"  {key}: {val}")

    # Visualizations
    figs = {
        'optimization_history': plot_optimization_history(study),
        'param_importances': plot_param_importances(study),
        'parallel_coordinate': plot_parallel_coordinate(study),
        'slice_plot': plot_slice(study),
        'contour_plot': plot_contour(study, params=['learning_rate', 'weight_decay'])
    }

    # Show or log
    for name, fig in figs.items():
        fig.show()

    # Also log to WandB
    _wandb = wandb
    _wandb.init(project=PROJECT_NAME, name="optuna_visualizations")
    _wandb.log(figs)
    _wandb.finish()
