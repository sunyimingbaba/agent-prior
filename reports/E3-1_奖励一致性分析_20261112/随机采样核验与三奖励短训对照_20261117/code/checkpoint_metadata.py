"""Read tensor/RNG checkpoint data without installing Lightning on the audit host."""
import pickle
import types
import torch


class _ModelCheckpointKey:
    """Inert dictionary-key marker; callback implementation is not executed."""
    pass


class _MetadataUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if (module, name) == ('pytorch_lightning.callbacks.model_checkpoint', 'ModelCheckpoint'):
            return _ModelCheckpointKey
        return super().find_class(module, name)


_PICKLE = types.SimpleNamespace(__name__='checkpoint_metadata_pickle',
    Unpickler=_MetadataUnpickler, load=pickle.load, loads=pickle.loads)


def load_checkpoint(path):
    return torch.load(path, map_location='cpu', weights_only=False, mmap=True, pickle_module=_PICKLE)
