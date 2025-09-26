import tensorflow as tf

class PrintLR(tf.keras.callbacks.Callback):
    """A custom callback to print the learning rate at the end of each epoch."""
    def on_epoch_end(self, epoch, logs=None):
        optimizer = self.model.optimizer
        
        current_lr = optimizer.learning_rate
        
        if isinstance(current_lr, tf.keras.optimizers.schedules.LearningRateSchedule):
            step = optimizer.iterations
            lr = current_lr(step)
            print(f" - LR: {lr.numpy():.7f}") 
        else:
            print(f" - LR: {current_lr:.7f}")

class WarmupCosineDecay(tf.keras.optimizers.schedules.LearningRateSchedule):
    """A schedule that implements warmup and cosine decay."""
    def __init__(self, initial_learning_rate, warmup_steps, decay_steps, alpha=0.0):
        super(WarmupCosineDecay, self).__init__()
        self.initial_learning_rate = initial_learning_rate
        self.warmup_steps = warmup_steps
        self.decay_steps = decay_steps
        self.alpha = alpha
        
        self.cosine_decay = tf.keras.optimizers.schedules.CosineDecay(
            initial_learning_rate=initial_learning_rate,
            decay_steps=decay_steps,
            alpha=alpha
        )

    def __call__(self, step):
        with tf.name_scope("WarmupCosineDecay"):
            initial_learning_rate = tf.convert_to_tensor(self.initial_learning_rate, name="initial_learning_rate")
            
            return tf.cond(
                step < self.warmup_steps,
                # Warmup Phase: Linear increase
                lambda: (initial_learning_rate / self.warmup_steps) * step,
                # Decay Phase: Cosine decay
                lambda: self.cosine_decay(step - self.warmup_steps)
            )

    def get_config(self):
        return {
            "initial_learning_rate": self.initial_learning_rate,
            "warmup_steps": self.warmup_steps,
            "decay_steps": self.decay_steps,
            "alpha": self.alpha,
        }