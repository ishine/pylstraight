# ------------------------------------------------------------------------ #
# Copyright 2025 Takenori Yoshimura                                        #
#                                                                          #
# Licensed under the Apache License, Version 2.0 (the "License");          #
# you may not use this file except in compliance with the License.         #
# You may obtain a copy of the License at                                  #
#                                                                          #
#     http://www.apache.org/licenses/LICENSE-2.0                           #
#                                                                          #
# Unless required by applicable law or agreed to in writing, software      #
# distributed under the License is distributed on an "AS IS" BASIS,        #
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. #
# See the License for the specific language governing permissions and      #
# limitations under the License.                                           #
# ------------------------------------------------------------------------ #

from __future__ import annotations

import os

import numpy as np

import pylstraight as pyls


def test_sample_with_debug_mode(sample_data: tuple[np.ndarray, int]) -> None:
    """Test sample with debug mode."""
    x, fs = sample_data
    os.environ["PYLSTRAIGHT_DEBUG"] = "1"
    f0 = pyls.fromfile("tests/reference/data.f0")
    ref_ap = pyls.sp_to_sp(pyls.fromfile("tests/reference/data.ap", fs), "db", "linear")
    hyp_ap = pyls.extract_ap(x, fs, f0)
    assert np.allclose(ref_ap, hyp_ap, atol=1e-2, rtol=0)


def test_sample_without_debug_mode(sample_data: tuple[np.ndarray, int]) -> None:
    """Test sample without debug mode."""
    x, fs = sample_data
    os.environ["PYLSTRAIGHT_DEBUG"] = "0"
    f0 = pyls.fromfile("tests/reference/data.f0")
    ref_ap = pyls.fromfile("tests/reference/data.ap", fs)
    ref_ap = pyls.sp_to_sp(ref_ap, "db", "linear")
    hyp_ap = pyls.extract_ap(x, fs, f0)
    # The error mainly arises from the random number generation.
    mean_error = np.abs(ref_ap - hyp_ap).mean()
    assert mean_error < 5e-3


def test_using_auxout(sample_data: tuple[np.ndarray, int]) -> None:
    """Test using the auxiliary output from the f0 extraction."""
    x, fs = sample_data
    f0, aux = pyls.extract_f0(x, fs, return_aux=True)
    ap = pyls.extract_ap(x, fs, f0)
    ap2 = pyls.extract_ap(x, fs, f0, aux)
    mean_error = np.abs(ap - ap2).mean()
    assert mean_error < 5e-3


def test_all_zero_input() -> None:
    """Test all zero input."""
    x, fs = np.zeros(8000), 8000
    f0 = np.zeros(200)
    ap = pyls.extract_ap(x, fs, f0)
    assert 0.9 < np.max(ap)


def test_short_f0_input() -> None:
    """Test short f0 input."""
    x, fs = np.zeros(8000), 8000
    f0 = np.zeros(199)
    ap = pyls.extract_ap(x, fs, f0)
    assert len(ap) == 199


def test_long_f0_input() -> None:
    """Test long f0 input."""
    x, fs = np.zeros(8000), 8000
    f0 = np.zeros(201)
    ap = pyls.extract_ap(x, fs, f0)
    assert len(ap) == 201


def test_conversion() -> None:
    """Test conversion between different aperiodicity formats."""

    def check_reversibility(x: np.ndarray, in_format: str, out_format: str) -> bool:
        """Check if the conversion is identity.

        Parameters
        ----------
        x : np.ndarray
            The input.

        in_format : str
            The input format.

        out_format : str
            The output format.

        Returns
        -------
        out : bool
            True if the conversion is identity.

        """
        y = pyls.ap_to_ap(x, in_format, out_format)
        z = pyls.ap_to_ap(y, out_format, in_format)
        return np.allclose(x, z)

    a = np.array([0.001, 0.5, 0.999])
    assert check_reversibility(a, "a", "p")
    assert check_reversibility(a, "a", "a/p")
    assert check_reversibility(a, "a", "p/a")
    p = pyls.ap_to_ap(a, "a", "p")
    assert check_reversibility(p, "p", "a/p")
    assert check_reversibility(p, "p", "p/a")
    a_p = pyls.ap_to_ap(p, "p", "a/p")
    assert check_reversibility(a_p, "a/p", "p/a")
