from rainbow_test.main_rainbow_test import k_fold_test
from models.LSTMModel import LSTMModel
from models.RainbowModel import RainbowModel
import utils.settings as settings
settings.init()

outputs = len(settings.ACTIVITIES)
models: list[RainbowModel] = [LSTMModel(window_size=180, stride_size=180, test_percentage=0.3, n_features=15, n_outputs=outputs, epochs=2), LSTMModel(window_size=180, stride_size=180, test_percentage=0.3, n_features=15, n_outputs=outputs, epochs=1)]
k_fold_test(models, ["epoch 2", "epoch 1"], "epoch_comparison", "we compare epochs", telegram=True)