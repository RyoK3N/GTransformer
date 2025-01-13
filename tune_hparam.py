import optuna
import os
import torch
import logging
import argparse
import pickle  
from train import train, parse_args
from optuna.trial import TrialState

def setup_logging(log_dir: str):
    """
    Set up logging configuration.
    Logs are saved to a file and also output to the console.
    """
    os.makedirs(log_dir, exist_ok=True)
    logging.basicConfig(
        filename=os.path.join(log_dir, 'finetune.log'),
        filemode='a',
        format='%(asctime)s - %(levelname)s - %(message)s',
        level=logging.INFO
    )

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)

def objective(trial):
    """
    Objective function for hyperparameter tuning with Optuna.
    """
    #  hyperparameters to tune
    random_seed = trial.suggest_int('random_seed', 42, 999)
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-2)
    weight_decay = trial.suggest_loguniform('weight_decay', 1e-6, 1e-3)
    batch_size = trial.suggest_categorical('batch_size', [256, 512])  
    warmup_epochs = trial.suggest_int('warmup_epochs', 2, 10)
    embed_dim = trial.suggest_categorical('embed_dim', [128, 256, 512])
    depth = trial.suggest_int('depth', 2, 6)
    num_heads = trial.suggest_categorical('num_heads', [4, 8, 16])
    drop_rate = trial.suggest_uniform('drop_rate', 0.1, 0.5)
    data_fraction = trial.suggest_uniform('data_fraction', 0.05, 0.5)  #  data_fraction

    args = parse_args()
    args.random_seed = random_seed
    args.learning_rate = learning_rate
    args.weight_decay = weight_decay
    args.batch_size = batch_size
    args.warmup_epochs = warmup_epochs
    args.embed_dim = embed_dim
    args.depth = depth
    args.num_heads = num_heads
    args.drop_rate = drop_rate
    args.data_fraction = data_fraction  
    args.log_dir = os.path.join('optuna_logs', f'trial_{trial.number}')
    args.model_save_path = os.path.join('optuna_weights', f'model_trial_{trial.number}.pth')

    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs('optuna_weights', exist_ok=True)

    # Set up logging for this trial
    logger = logging.getLogger(f'Trial_{trial.number}')
    handler = logging.FileHandler(os.path.join(args.log_dir, 'trial.log'))
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    if not logger.handlers:
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)

    try:
        logger.info(f"Starting Trial {trial.number}")
        logger.info(
            f"Hyperparameters: random_seed={random_seed}, learning_rate={learning_rate}, "
            f"weight_decay={weight_decay}, batch_size={batch_size}, warmup_epochs={warmup_epochs}, "
            f"embed_dim={embed_dim}, depth={depth}, num_heads={num_heads}, drop_rate={drop_rate}, "
            f"data_fraction={data_fraction}"
        )

        # Call train function
        train(args)

        # After training, load the validation loss from the saved model
        if os.path.exists(args.model_save_path):
            checkpoint = torch.load(args.model_save_path, map_location='cpu')
            val_loss = checkpoint.get('loss', float('inf'))
            logger.info(f"Trial {trial.number} completed with validation loss: {val_loss}")
        else:
            val_loss = float('inf')
            logger.warning(f"Trial {trial.number} did not save a model. Assigning val_loss as infinity.")

    except Exception as e:
        logger.error(f"Error during trial {trial.number}: {e}", exc_info=True)
        val_loss = float('inf')

    return val_loss

def main():
    """
    Main function to run the Optuna optimization.
    """
    parser = argparse.ArgumentParser(description='Hyperparameter tuning with Optuna for KTPFormer.')
    parser.add_argument('--study_name', type=str, default='ktpformer_hyperparameter_optimization',
                        help='Name of the Optuna study.')
    parser.add_argument('--n_trials', type=int, default=20,
                        help='Number of trials for optimization.')
    parser.add_argument('--timeout', type=int, default=3600,
                        help='Maximum time in seconds for the optimization.')
    parser.add_argument('--log_dir', type=str, default='optuna_logs',
                        help='Directory to save optimization logs.')
    parser.add_argument('--weight_dir', type=str, default='optuna_weights',
                        help='Directory to save model weights.')
    parser.add_argument('--study_file', type=str, default='optuna_study_results/optuna_study.pkl',
                        help='File path to save/load the Optuna study.')
    args = parser.parse_args()

    # logging
    setup_logging(args.log_dir)
    logger = logging.getLogger('Finetune')

    logger.info("Starting Optuna hyperparameter tuning.")
    logger.info(f"Study Name: {args.study_name}")
    logger.info(f"Number of Trials: {args.n_trials}")
    logger.info(f"Timeout: {args.timeout} seconds")
    logger.info(f"Study File: {args.study_file}")

    # Fpath for the Optuna study as a pickle file
    study_path = args.study_file
    study_dir = os.path.dirname(study_path)
    os.makedirs(study_dir, exist_ok=True)

    if os.path.exists(study_path):
        try:
            with open(study_path, "rb") as f:
                study = pickle.load(f)
            logger.info(f"Loaded existing study from {study_path}")
        except Exception as e:
            logger.error(f"Failed to load study from {study_path}: {e}")
            study = optuna.create_study(
                study_name=args.study_name,
                direction="minimize",
                sampler=optuna.samplers.TPESampler(seed=42),
                pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10)
            )
            logger.info("Created a new study.")
    else:
        study = optuna.create_study(
            study_name=args.study_name,
            direction="minimize",
            sampler=optuna.samplers.TPESampler(seed=42),
            pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10)
        )
        logger.info("Created a new study.")

    # Configure resource limits
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    torch.set_num_threads(2)  # Limit CPU thread to 2

    try:
        study.optimize(
            objective,
            n_trials=args.n_trials,
            timeout=args.timeout,
            gc_after_trial=True,
            show_progress_bar=True
        )
    except KeyboardInterrupt:
        logger.warning("Optimization interrupted manually.")

    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    logger.info("Study statistics:")
    logger.info(f"  Number of finished trials: {len(study.trials)}")
    logger.info(f"  Number of pruned trials: {len(pruned_trials)}")
    logger.info(f"  Number of complete trials: {len(complete_trials)}")

    if study.best_trial:
        logger.info("\nOptimization finished.")
        logger.info(f"Best trial: {study.best_trial.number}")
        logger.info(f"Best value (validation loss): {study.best_trial.value:.4f}")
        logger.info("Best hyperparameters:")
        for key, value in study.best_trial.params.items():
            logger.info(f"  {key}: {value}")
    else:
        logger.warning("No trials completed successfully.")

    # Save the study for future use using pickle
    try:
        with open(study_path, "wb") as f:
            pickle.dump(study, f)
        logger.info(f"Study saved to {study_path}")
    except Exception as e:
        logger.error(f"Failed to save study to {study_path}: {e}")

if __name__ == "__main__":
    main()
