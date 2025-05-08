import optuna
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from data.feature_dataset import FeatureDataset
from models.videollava import VideoLlava
from utils import load_model
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
from argparse import ArgumentParser
import os

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
PROJECT_NAME = "NewbornActivityRecognition"


def objective(trial: optuna.Trial) -> float:
    
    logger.info(f"Trial number: {trial.number}")
    wandb.init(
        project=PROJECT_NAME,
        reinit=True,
        config={},
        name=f"optim_trial_{trial.number}"
    )
    # Sample hyperparameters
    lr = trial.suggest_loguniform('learning_rate', 1e-6, 1e-2)
    weight_decay = trial.suggest_loguniform('weight_decay', 1e-6, 1e-2)
    optimizer_name = trial.suggest_categorical('optimizer', ['adamw', 'sgd', 'adam'])
    dropout_rate = trial.suggest_uniform('dropout', 0.0, 0.5)
    threshold = trial.suggest_float('threshold', 0.3, 0.7)
    hidden_dim = trial.suggest_categorical('hidden_dim', [256, 512, 1024])
    if optimizer_name in ('sgd', 'adamw'):
        momentum = trial.suggest_uniform('momentum', 0.0, 0.99)
    else:
        momentum = None
    epochs = trial.suggest_int('epochs', 3, 21, step=2)
    patience = trial.suggest_int('patience', 3, 8)
    scheduler_name = trial.suggest_categorical(
        'scheduler_name', ['steplr', 'cosineannealinglr', 'reduceonplateau']
    )
    scheduler_patience = trial.suggest_int('scheduler_patience', 1, 5)
    

    # Initialize model
    model = load_model(model_name = model_name, checkpoint = None)
    # Rebuild classifier with sampled architecture
    num_classes = model.classifier[-1].out_features
    backbone_hidden = model.backbone.config.text_config.hidden_size
    model.classifier = nn.Sequential(
        nn.Linear(backbone_hidden, hidden_dim),
        nn.ReLU(),
        nn.Dropout(dropout_rate),
        nn.Linear(hidden_dim, num_classes)
    ).to(DEVICE)

    # Freeze backbone, train only classifier
    model.set_freezing_condition('all')

    # Define criterion and optimizer
    criterion = model.define_criterion('bce')
    optimizer = model.define_optimizer(
        optimizer_name=optimizer_name,
        learning_rate=lr,
        momentum=momentum,
        weight_decay=weight_decay
    )

    # Optionally define a scheduler
    scheduler = model.define_scheduler(
        scheduler_name=scheduler_name,
        optimizer=optimizer,
        epochs=epochs,
        patience=scheduler_patience,
        step_size=trial.suggest_int('step_size', 2, 10),
        gamma=trial.suggest_float('gamma', 0.1, 0.9)
    )
    
    wandb.config.update({
        'learning_rate': lr,
        'weight_decay': weight_decay,
        'optimizer_name': optimizer_name,
        'dropout_rate': dropout_rate,
        'threshold': threshold,
        'hidden_dim': hidden_dim,
        'momentum': momentum,
        'epochs': epochs,
        'patience': patience,
        'scheduler_name': scheduler_name,
        'scheduler_patience': scheduler_patience
    })

    # Train classifier head
    results = model.train_classifier(
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        epochs=epochs,
        optimizer=optimizer,
        criterion=criterion,
        threshold = threshold,
        patience=patience,
        scheduler=scheduler,
        show_progress=False
    )

    # Return the validation macro F1 score
    val_metrics = results['val_metrics']
    val_f1 = val_metrics['f1_macro']
    wandb.log({'val_f1_macro': val_f1})
    logger.info(f"Trial {trial.number} - Validation F1: {val_f1:.4f}")
    wandb.finish()
    return val_f1
    

if __name__ == '__main__':
    
    parser = ArgumentParser()
    parser.add_argument('--model_name', type=str, default='videollava', help='Model name')
    args = parser.parse_args()
    model_name = args.model_name
    
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    logger.info("Starting Optuna study with persistence.")
    
    DB_PATH = f"sqlite:///hp_opt/{model_name}/optuna_newborn.db"  # Path to persistent storage
    STUDY_NAME = "newborn_activity_recognition"
    os.makedirs(os.path.dirname(f"hp_opt/{model_name}/optuna_newborn.db"), exist_ok=True)
    
    folder = "2sec_4fps"
    train_dataset = FeatureDataset(f'data/features/{model_name}/train/{folder}')
    val_dataset = FeatureDataset(f'data/features/{model_name}/validation/{folder}')
    train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True,
                                  num_workers=1, drop_last = True)
    val_dataloader = DataLoader(val_dataset, batch_size=8, shuffle=False,
                                num_workers=1, drop_last = True)

    
    study = optuna.create_study(
        study_name=STUDY_NAME,
        storage=DB_PATH,
        direction='maximize',
        sampler=optuna.samplers.TPESampler(),
        pruner=optuna.pruners.MedianPruner(),
        load_if_exists=True
    )
    
    wandb_callback = WeightsAndBiasesCallback(
        metric_name="val_f1_macro",
        wandb_kwargs={"project": PROJECT_NAME, "reinit": True}
    )
    
    study.optimize(objective, n_trials=None, n_jobs=1, callbacks=[wandb_callback])
    
    best = study.best_trial
    logger.info(f"Best trial: #{best.number} Value: {best.value:.4f}")
    logger.info("Params:")
    for key, val in best.params.items():
        logger.info(f"  {key}: {val}")

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
