SEED_OFFSET = 28
VAL_PERCENT = 0.1
TEST_PERCENT = 0.1
MAX_SEQ_LENGTH = 512

SYSTEM_CONFIG = {'per_device_train_batch_size': 32,
                 'gradient_accumulation_steps': 4}

# logging_steps: logging when training roberta
# the frequency of evaluating on the val set for roberta
TRAIN_CONFIG = {'epoch': 2,
                'logging_steps': 20,
                'eval_steps': 50}

TRAIN_DEBUG_CONFIG = {'epoch': 1,
                      'logging_steps': 1,
                      'eval_steps': 2}
