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

import numpy as np

import pylstraight as pyls


def test_synthesis() -> None:
    """Test sample data given in the reference parameters."""
    ref_syn, fs = pyls.read("tests/reference/data.syn.wav")
    f0 = pyls.fromfile("tests/reference/data.f0")
    ap = pyls.fromfile("tests/reference/data.ap", fs)
    ap = pyls.sp_to_sp(ap, "db", "linear")
    sp = pyls.fromfile("tests/reference/data.sp", fs)
    hyp_syn = pyls.synthesize(f0, ap, sp, fs)[: len(ref_syn)]
    assert 0.95 < np.corrcoef(ref_syn, hyp_syn)[0, 1]
    pyls.write("tests/output/data.syn.wav", hyp_syn, fs)


def test_sample(sample_data: tuple[np.ndarray, int]) -> None:
    """Test sample data."""
    ref_syn, fs = pyls.read("tests/reference/data.syn.wav")
    x, _ = sample_data
    f0 = pyls.extract_f0(x, fs)
    ap = pyls.extract_ap(x, fs, f0)
    sp = pyls.extract_sp(x, fs, f0)
    hyp_syn = pyls.synthesize(f0, ap, sp, fs)[: len(ref_syn)]
    assert 0.95 < np.corrcoef(ref_syn, hyp_syn)[0, 1]
    pyls.write("tests/output/data.pyls.wav", hyp_syn, fs)


def test_vaiueo() -> None:
    """Test another sample data."""
    x, fs = pyls.read("tools/straight/src/vaiueo2d.wav")
    fp = 1
    f0 = pyls.extract_f0(x, fs, frame_shift=fp)
    ap = pyls.extract_ap(x, fs, f0, frame_shift=fp)
    sp = pyls.extract_sp(x, fs, f0, frame_shift=fp)
    syn = pyls.synthesize(f0, ap, sp, fs, frame_shift=fp)
    pyls.write("tests/output/vaiueo.pyls.wav", syn, fs)


def test_all_zero_input() -> None:
    """Test all zero input."""
    fs = 8000
    f0 = np.zeros(200)
    ap = np.zeros((200, 513)) + 1e-8
    sp = np.zeros((200, 513)) + 1e-8
    syn = pyls.synthesize(f0, ap, sp, fs)
    assert len(syn) == 8000


def test_short_f0_input() -> None:
    """Test short f0 input."""
    fs = 8000
    f0 = np.ones(1) * 40
    ap = np.ones((200, 513))
    sp = np.ones((200, 513))
    syn = pyls.synthesize(f0, ap, sp, fs)
    assert len(syn) == 40


def test_short_sp_input() -> None:
    """Test short sp input."""
    fs = 8000
    f0 = np.ones(200) * 40
    ap = np.ones((1, 513))
    sp = np.ones((1, 513))
    syn = pyls.synthesize(f0, ap, sp, fs)
    assert len(syn) == 40


def test_short_fftl_input() -> None:
    """Test short FFT length."""
    fs = 8000
    f0 = np.ones(200) * 40
    ap = np.ones((200, 400))
    sp = np.ones((200, 513))
    syn = pyls.synthesize(f0, ap, sp, fs)
    assert len(syn) == 8000
    syn = pyls.synthesize(f0, ap, sp[:, :400], fs)
    assert len(syn) == 8000
