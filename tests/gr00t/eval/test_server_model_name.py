# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import MagicMock, patch

from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.eval.run_gr00t_server import ServerConfig, main
import tyro


def test_server_config_parses_model_name_override():
    config = tyro.cli(
        ServerConfig,
        args=[
            "--model-path",
            "/models/tianji",
            "--model-name",
            "/models/Cosmos-Reason2-2B",
        ],
    )

    assert config.model_name == "/models/Cosmos-Reason2-2B"


def test_server_forwards_model_name_override(tmp_path):
    model_name = "/models/Cosmos-Reason2-2B"
    server = MagicMock()
    server.__enter__.return_value = server

    with (
        patch("gr00t.eval.run_gr00t_server.Gr00tPolicy") as mock_policy,
        patch("gr00t.eval.run_gr00t_server.PolicyServer", return_value=server),
    ):
        main(
            ServerConfig(
                model_path=str(tmp_path),
                model_name=model_name,
                device="cpu",
            )
        )

    mock_policy.assert_called_once_with(
        embodiment_tag=EmbodimentTag.NEW_EMBODIMENT,
        model_path=str(tmp_path),
        device="cpu",
        model_name=model_name,
        strict=True,
    )
    server.run.assert_called_once_with()
