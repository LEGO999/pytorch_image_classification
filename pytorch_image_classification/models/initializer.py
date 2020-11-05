from typing import Callable

import torch.nn as nn


def create_initializer(mode: str) -> Callable:
    if mode in ['kaiming_fan_out', 'kaiming_fan_in']:
        # key_word in kaiming_normal_ is 'fan_out' or 'fan_in'
        # 'fan_out' could keep the magnitude of backward gradient
        mode = mode[8:]

        # initializer initializes the weights and biases of the module
        def initializer(module):
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight.data,
                                        mode=mode,
                                        nonlinearity='relu')
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight.data)
                nn.init.zeros_(module.bias.data)
            elif isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight.data,
                                        mode=mode,
                                        nonlinearity='relu')
                nn.init.zeros_(module.bias.data)
    else:
        raise ValueError()

    return initializer
