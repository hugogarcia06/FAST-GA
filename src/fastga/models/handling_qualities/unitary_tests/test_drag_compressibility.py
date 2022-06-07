#  This file is part of FAST-OAD : A framework for rapid Overall Aircraft Design
#  Copyright (C) 2022 ONERA & ISAE-SUPAERO
#  FAST is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#  You should have received a copy of the GNU General Public License
#  along with this program.  If not, see <https://www.gnu.org/licenses/>.
import pytest
from fastga.models.handling_qualities.utils.drag_compressibility import *


def test_drag_compressibility():

    assert get_drag_compressibility(0.3) == pytest.approx(0.0, abs=0.001)
    assert get_drag_compressibility(0.6) == pytest.approx(0.0, abs=0.001)
    assert get_drag_compressibility(0.8) == pytest.approx(0.005, abs=0.001)
    assert get_drag_compressibility(1.0) == pytest.approx(0.025, abs=0.001)
    assert get_drag_compressibility(1.2) == pytest.approx(0.03, abs=0.001)

def test_drag_mach_derivative():

    # assert get_drag_mach_derivative(0.3) == pytest.approx(0.0, abs=0.001)
    # assert get_drag_mach_derivative(0.6) == pytest.approx(0.0, abs=0.001)
    # assert get_drag_mach_derivative(0.9) == pytest.approx(0.005, abs=0.001)
    # assert get_drag_mach_derivative(1.0) == pytest.approx(0.025, abs=0.001)
    assert get_drag_mach_derivative(1.2) == pytest.approx(0.03, abs=0.001)
