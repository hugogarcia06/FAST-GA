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
import numpy as np
import os.path as pth

from fastga.models.handling_qualities.unitary_tests.test_functions import comp_stab_coef
from fastga.models.handling_qualities.external.openvsp.openvsp import OPENVSPSimpleGeometry
from fastga.models.handling_qualities.unitary_tests import data

XML_FILE = "beechcraft_76.xml"
TEST_STAB_FILE = "Test_Stab_file.stab"


def test_openvsp_comp_stab_coef():
    """
    Tests openvsp stability coefficients computation
    Data:
    - AoA = 2.0 deg
    - Altitude = 10000 meters
    - Mach = 0.2
    """

    # Check values
    cL_u = 0.0104104
    cD_u = 0.0002405
    cm_u = -0.0009006
    cL_alpha = 4.9402129
    cD_alpha = 0.1395310
    cm_alpha = -1.6575748
    cL_q = 10.1444559
    cD_q = 0.4052255
    cm_q = -18.4688821
    cY_beta = 0.0071155
    cl_beta = -0.0143223
    cn_beta = -0.0010073
    cY_p = 0.0007226
    cl_p = -0.4842667
    cn_p = -0.0467423
    cY_r = -0.0006357
    cl_r = 0.0581296
    cn_r = 0.0006449

    comp_stab_coef(
        XML_FILE,
        cL_u,
        cD_u,
        cm_u,
        cL_alpha,
        cD_alpha,
        cm_alpha,
        cL_q,
        cD_q,
        cm_q,
        cY_beta,
        cl_beta,
        cn_beta,
        cY_p,
        cl_p,
        cn_p,
        cY_r,
        cl_r,
        cn_r
    )


def test_read_Stab_file():
    file = pth.join(data.__path__[0], TEST_STAB_FILE)
    (
        cL_u,
        cD_u,
        cm_u,
        cL_alpha,
        cD_alpha,
        cm_alpha,
        cL_q,
        cD_q,
        cm_q,
        cY_beta,
        cl_beta,
        cn_beta,
        cY_p,
        cl_p,
        cn_p,
        cY_r,
        cl_r,
        cn_r
    ) = OPENVSPSimpleGeometry.read_stab_file(file)

    assert cL_u == pytest.approx(0.0418305, rel=0.001)
    assert cD_u == pytest.approx(0.0016583, rel=0.001)
    assert cm_u == pytest.approx(0.0038588, rel=0.001)
    assert cL_alpha == pytest.approx(5.8677326, rel=0.001)
    assert cD_alpha == pytest.approx(0.2314843, rel=0.001)
    assert cm_alpha == pytest.approx(-3.9592064, rel=0.001)
    assert cL_q == pytest.approx(14.8667623, rel=0.001)
    assert cD_q == pytest.approx(-1.0520681, rel=0.001)
    assert cm_q == pytest.approx(-59.5380225, rel=0.001)
    assert cY_beta == pytest.approx(-0.4774181, rel=0.001)
    assert cl_beta == pytest.approx(-0.0978689, rel=0.001)
    assert cn_beta == pytest.approx(0.2221262, rel=0.001)
    assert cY_p == pytest.approx(-0.0675308, rel=0.001)
    assert cl_p == pytest.approx(-0.6163760, rel=0.001)
    assert cn_p == pytest.approx(-0.0795473, rel=0.001)
    assert cY_r == pytest.approx(0.5207040, rel=0.001)
    assert cl_r == pytest.approx(0.1994534, rel=0.001)
    assert cn_r == pytest.approx(-0.2353378, rel=0.001)


