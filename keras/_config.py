__all__ = [
    'KerasConfig',
    'DEFAULT_MODEL_NAME',
]

DEFAULT_MODEL_NAME = 'model.h5'


class KerasConfig:
    def __init__(self,
                 epoch,
                 batch_size,
                 callbacks=None,
                 model_name: str = DEFAULT_MODEL_NAME,
                 print_summary: bool = True,
                 save_model: bool = True
                 ) -> None:
        if callbacks is None:
            callbacks = []

        self.epoch = epoch
        self.batch_size = batch_size
        self.callbacks = callbacks
        self.model_name = model_name
        self.print_summary = print_summary
        self.save_model = save_model
