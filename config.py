# Configuration parameters
RANDOM_SEED = 42
FAIRNESS_THRESHOLD_DI = 0.8
FAIRNESS_THRESHOLD_EOD = 0.1
INTERVENTION_RATIO = 0.1  # 10% samples trigger human intervention
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Model hyperparameters
LR_EPOCHS = 100
DL_EPOCHS = 50
BATCH_SIZE = 128
LEARNING_RATE = 0.001