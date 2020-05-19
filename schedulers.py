import tensorflow as tf


class CircularLearningRate(tf.keras.callbacks.Callback):
    def __init__(self, min_lr_1, max_lr, min_lr_2):
        super().__init__()
        self.min_lr_1 = min_lr_1
        self.max_lr = max_lr
        self.min_lr_2 = min_lr_2

    def set_dataset(self, data, batch_size):
        self.mid = len(data) / batch_size / 2

    def on_train_batch_end(self, batch, logs=None):
        if batch < self.mid:
            lb = self.min_lr_1
            way = self.mid - batch
        else:
            lb = self.min_lr_2
            way = batch - self.mid
        lr = self.max_lr - way / self.mid * (self.max_lr - lb)
        tf.keras.backend.set_value(self.model.optimizer.lr, lr)


class CircularLearningRateSched(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, min_lr_1, max_lr, min_lr_2, mid=1000):
        super().__init__()
        self.min_lr_1 = min_lr_1
        self.max_lr = max_lr
        self.min_lr_2 = min_lr_2
        self.mid = mid
    
    @tf.function
    def __call__(self, step):
        if step < self.mid:
            lb = self.min_lr_1
            way = self.mid - step
        else:
            lb = self.min_lr_2
            way = step - self.mid
        return self.max_lr - way / self.mid * (self.max_lr - lb)
