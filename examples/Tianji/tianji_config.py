# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from gr00t.configs.data.embodiment_configs import register_modality_config
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.types import (
    ActionConfig,
    ActionFormat,
    ActionRepresentation,
    ActionType,
    ModalityConfig,
)


_absolute_vector = ActionConfig(
    rep=ActionRepresentation.ABSOLUTE,
    type=ActionType.NON_EEF,
    format=ActionFormat.DEFAULT,
)

tianji_pico_config = {
    "video": ModalityConfig(
        delta_indices=[0],
        modality_keys=["cam_high", "cam_left_wrist", "cam_right_wrist"],
    ),
    "state": ModalityConfig(
        delta_indices=[0],
        modality_keys=["tianji_ee"],
    ),
    "action": ModalityConfig(
        delta_indices=list(range(40)),
        modality_keys=["tianji_ee"],
        # The vector already uses OpenPI/robocoin's column-rot6d convention.
        # Keep the full 20D contract opaque and absolute so GR00T does not reinterpret rot6d.
        action_configs=[_absolute_vector],
    ),
    "language": ModalityConfig(
        delta_indices=[0],
        modality_keys=["annotation.human.task_description"],
    ),
}

register_modality_config(tianji_pico_config, embodiment_tag=EmbodimentTag.NEW_EMBODIMENT)
