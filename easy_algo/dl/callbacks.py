import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from enum import Enum


class CallbackType(Enum):
    EARLY_STOPPING = 'early_stopping'
    MODEL_CHECKPOINT = 'model_checkpoint'
    REDUCE_LR_ON_PLATEAU = 'reduce_lr_on_plateau'


class CallbackBuilder:
    def __init__(self):
        self.callbacks = []

    def add_early_stopping(self, monitors=None, min_delta=0, patience=0, verbose=0, mode='auto', baseline=None,
                           restore_best_weights=False):
        """
        添加EarlyStopping回调。
        """
        if monitors is None:
            monitors = ['val_loss']
        self.callbacks.append(
            EarlyStopping(monitor=monitors, min_delta=min_delta, patience=patience, verbose=verbose, mode=mode,
                          baseline=baseline, restore_best_weights=restore_best_weights))

    def add_model_checkpoint(self, filepath, monitor='val_loss', verbose=0, save_best_only=False,
                             save_weights_only=False, mode='auto', period=1):
        """
        添加ModelCheckpoint回调。
        """
        self.callbacks.append(ModelCheckpoint(filepath, monitor=monitor, verbose=verbose, save_best_only=save_best_only,
                                              save_weights_only=save_weights_only, mode=mode, period=period))

    def add_reduce_lr_on_plateau(self, monitor='val_loss', factor=0.1, patience=10, verbose=0, mode='auto',
                                 min_delta=0.0001, cooldown=0, min_lr=0):
        """
        添加ReduceLROnPlateau回调。
        """
        self.callbacks.append(
            ReduceLROnPlateau(monitor=monitor, factor=factor, patience=patience, verbose=verbose, mode=mode,
                              min_delta=min_delta, cooldown=cooldown, min_lr=min_lr))

    def build(self):
        """
        构建并返回所有配置好的回调。
        """
        return self.callbacks

    def _parse_callback_config(self, callback_type, config):
        if callback_type == CallbackType.EARLY_STOPPING:
            return EarlyStopping(**config)
        elif callback_type == CallbackType.MODEL_CHECKPOINT:
            return ModelCheckpoint(**config)
        elif callback_type == CallbackType.REDUCE_LR_ON_PLATEAU:
            return ReduceLROnPlateau(**config)
        else:
            raise ValueError("Unknown callback type")

    def build_from_json(self, config):
        """
        从JSON字符串构建回调。
        """
        for callback_type, params in config.items():
            callback = self._parse_callback_config(CallbackType[callback_type], params)
            self.callbacks.append(callback)
        return self.callbacks

