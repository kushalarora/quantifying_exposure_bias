from allennlp.common.registrable import Registrable
from allennlp.data.vocabulary import Vocabulary
import torch

class Oracle(Registrable):
    def __init__(self, 
                 num_samples : int):
        pass

    def sample_training_set(self):
        return NotImplementedError

    def compute_sent_probs(self, sentences):
        return NotImplementedError

    def reference_rollout(self, 
                            prediction_prefixes: torch.LongTensor, 
                            rollout_steps: int,
                            vocab: Vocabulary,
                            ):
        return NotImplementedError

