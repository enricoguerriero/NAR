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

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PROJECT_NAME = "NewbornActivityRecognition"

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
    # Sample hyperparameters
    optimizer_name = trial.suggest_categorical('optimizer_name', ['adam', 'sgd', 'adamw'])
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-2)
    weight_decay = trial.suggest_loguniform('weight_decay', 1e-6, 1e-2)
    epochs = trial.suggest_int('epochs', 10, 50, step=5)
    momentum = None
    if optimizer_name == 'sgd' or optimizer_name == 'adamw':
        momentum = trial.suggest_uniform('momentum', 0.0, 0.99)

    criterion_name = "wbce"
    pos_weight = torch.tensor([0.19311390817165375, 2.532083511352539, 7.530612468719482, 6.510387420654297]).to(device=DEVICE)
    prior_probability = torch.tensor([0.8381429314613342, 0.2831190228462219, 0.11722487956285477, 0.1331489235162735]).to(device=DEVICE)
    
    threshold = trial.suggest_float('threshold', 0.3, 0.7)

    scheduler_name = trial.suggest_categorical(
        'scheduler_name', ['steplr', 'cosineannealinglr', 'reduceonplateau']
    )
    
    scheduler_patience = trial.suggest_int('scheduler_patience', 1, 5)
    patience = trial.suggest_int('patience', 3, 8)
    
    freezing_condition = trial.suggest_categorical('freezing_condition', ['none', 'all', 'partial'])


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

    # Initialize model
    model = TimeSformer(device=DEVICE).to(DEVICE)

    # Run training
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
        logger = logger
    )

    # Objective: maximize validation macro F1
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
    
    wandb_callback = WeightsAndBiasesCallback(
        metric_name="val_f1_macro",
        wandb_kwargs={
            "project": PROJECT_NAME,
            "reinit": True
        }
    )
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    logger.info("Starting Optuna trial.")
    # Load datasets
    train_dataset = TokenDataset('data/tokens/TimeSformer/train/2sec_4fps')
    val_dataset = TokenDataset('data/tokens/TimeSformer/validation/2sec_4fps')
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=4, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=4, collate_fn=collate_fn)
    # Create study and optimize
    study = optuna.create_study(
        direction='maximize',
        sampler=optuna.samplers.TPESampler(),
        pruner=optuna.pruners.MedianPruner()
    )
    study.optimize(objective, n_trials=None, n_jobs=1, callbacks=[wandb_callback])

    # Print best results
    logger.info("Best trial:")
    logger.inf(f"  Value: {study.best_trial.value:.4f}")
    logger.info("  Params:")
    for key, val in study.best_trial.params.items():
        logger.info(f"    {key}: {val}")

    # Visualize the optimization
    fig1 = plot_optimization_history(study)
    fig2 = plot_param_importances(study)
    fig3 = plot_parallel_coordinate(study)
    fig4 = plot_slice(study)
    fig5 = plot_contour(study, params=['learning_rate', 'weight_decay'])

    # Optionally show figures
    fig1.show()
    fig2.show()
    fig3.show()
    fig4.show()
    fig5.show()

    # Or log the figures to WandB
    import wandb as _wandb
    _wandb.init(project=PROJECT_NAME, name="optuna_visualizations")
    _wandb.log({
        "optimization_history": fig1,
        "param_importances": fig2,
        "parallel_coordinate": fig3,
        "slice_plot": fig4,
        "contour_plot": fig5
    })
    _wandb.finish()
