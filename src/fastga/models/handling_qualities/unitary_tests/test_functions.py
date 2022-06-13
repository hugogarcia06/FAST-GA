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

import logging
import shutil
import glob
import time
import tempfile
import os
import os.path as pth
import pytest
import numpy as np

from platform import system
from pathlib import Path
from tempfile import TemporaryDirectory

from fastga.models.handling_qualities.external.openvsp.compute_stab import ComputeSTABopenvsp
from tests.testing_utilities import run_system, get_indep_var_comp, list_inputs
from tests.xfoil_exe.get_xfoil import get_xfoil_path

from fastga.models.aerodynamics.external.xfoil import resources


RESULTS_FOLDER = pth.join(pth.dirname(__file__), "results")
TMP_SAVE_FOLDER = "test_save"
xfoil_path = None if system() == "Windows" else get_xfoil_path()

_LOGGER = logging.getLogger(__name__)


def _create_tmp_directory() -> TemporaryDirectory:
    """Provide temporary directory for calculation!"""
    for tmp_base_path in [None, pth.join(str(Path.home()), ".fast")]:
        if tmp_base_path is not None:
            os.makedirs(tmp_base_path, exist_ok=True)
        tmp_directory = tempfile.TemporaryDirectory(prefix="x", dir=tmp_base_path)
        break

    return tmp_directory


def reshape_curve(y, cl):
    """Reshape data from openvsp/vlm lift curve!"""
    for idx in range(len(y)):
        if np.sum(y[idx : len(y)] == 0) == (len(y) - idx):
            y = y[0:idx]
            cl = cl[0:idx]
            break

    return y, cl


def reshape_polar(cl, cdp):
    """Reshape data from xfoil polar vectors!"""
    for idx in range(len(cl)):
        if np.sum(cl[idx : len(cl)] == 0) == (len(cl) - idx):
            cl = cl[0:idx]
            cdp = cdp[0:idx]
            break
    return cl, cdp


def polar_result_transfer():
    # Put saved polar results in a temporary folder to activate Xfoil run and have repeatable
    # results [need writing permission]

    tmp_folder = _create_tmp_directory()

    files = glob.iglob(pth.join(resources.__path__[0], "*.csv"))

    for file in files:
        if os.path.isfile(file):
            shutil.copy(file, tmp_folder.name)
            # noinspection PyBroadException
            try:
                os.remove(file)
            except:
                _LOGGER.info("Cannot remove %s file!" % file)

    return tmp_folder


def polar_result_retrieve(tmp_folder):
    # Retrieve the polar results set aside during the test duration if there are some [need
    # writing permission]

    files = glob.iglob(pth.join(tmp_folder.name, "*.csv"))

    for file in files:
        if os.path.isfile(file):
            # noinspection PyBroadException
            try:
                shutil.copy(file, resources.__path__[0])
            except (OSError, shutil.SameFileError) as e:
                if isinstance(e, OSError):
                    _LOGGER.info(
                        "Cannot copy %s file to %s! Likely due to permission error"
                        % (file, tmp_folder.name)
                    )
                else:
                    _LOGGER.info(
                        "Cannot copy %s file to %s! Likely because the file already exists in the "
                        "target directory " % (file, tmp_folder.name)
                    )

    tmp_folder.cleanup()


def compute_stab(
    add_fuselage: bool,
    XML_FILE: str,
    ):
    """Compute stability coefficients for the whole aircraft"""
    # Create result temporary directory
    results_folder = _create_tmp_directory()

    # Transfer saved polar results to temporary folder
    tmp_folder = polar_result_transfer()

    # Research independent input value in .xml file
    # noinspection PyTypeChecker
    ivc = get_indep_var_comp(
        list_inputs(ComputeSTABopenvsp()), __file__, XML_FILE
    )

    # Run problem twice
    start = time.time()
    # noinspection PyTypeChecker
    problem = run_system(
        ComputeSTABopenvsp(
            result_folder_path=results_folder.name,
            add_fuselage=add_fuselage
        ),
        ivc,
    )
    stop = time.time()
    duration_1st_run = stop - start
    start = time.time()
    # noinspection PyTypeChecker
    run_system(
        ComputeSTABopenvsp(
            result_folder_path=results_folder.name,
            add_fuselage=add_fuselage
        ),
        ivc,
    )
    stop = time.time()

    duration_2nd_run = stop - start

    # Retrieve polar results from temporary folder
    polar_result_retrieve(tmp_folder)

    # Remove existing result files
    results_folder.cleanup()

    # Check obtained value(s) is/(are) correct
    assert (duration_2nd_run / duration_1st_run) <= 0.1

    # Return problem for complementary values check
    return problem


def comp_stab_coef(
        add_fuselage,
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
):
    """Tests aircraft stability at high speed!"""
    problem = compute_stab(add_fuselage, XML_FILE)

    # Check obtained value(s) is/(are) correct
    assert problem["data:handling_qualities:longitudinal:derivatives:CL:speed"] == pytest.approx(
        cL_u, abs=1e-4
    )
    assert problem["data:handling_qualities:longitudinal:derivatives:CD:speed"] == pytest.approx(
        cD_u, abs=1e-4
    )
    assert problem["data:handling_qualities:longitudinal:derivatives:Cm:speed"] == pytest.approx(
        cm_u, abs=1e-4
    )
    assert problem["data:handling_qualities:longitudinal:derivatives:CL:alpha"] == pytest.approx(
        cL_alpha, abs=1e-4
    )
    assert problem["data:handling_qualities:longitudinal:derivatives:CD:alpha"] == pytest.approx(
        cD_alpha, abs=1e-4
    )
    assert problem["data:handling_qualities:longitudinal:derivatives:Cm:alpha"] == pytest.approx(
        cm_alpha, abs=1e-4
    )
    assert problem["data:handling_qualities:longitudinal:derivatives:CL:pitchrate"] == pytest.approx(
        cL_q, abs=1e-4
    )
    assert problem["data:handling_qualities:longitudinal:derivatives:CD:pitchrate"] == pytest.approx(
        cD_q, abs=1e-4
    )
    assert problem["data:handling_qualities:longitudinal:derivatives:Cm:pitchrate"] == pytest.approx(
        cm_q, abs=1e-4
    )
    assert problem["data:handling_qualities:lateral:derivatives:CY:beta"] == pytest.approx(
        cY_beta, abs=1e-4
    )
    assert problem["data:handling_qualities:lateral:derivatives:Cl:beta"] == pytest.approx(
        cl_beta, abs=1e-4
    )
    assert problem["data:handling_qualities:lateral:derivatives:Cn:beta"] == pytest.approx(
        cn_beta, abs=1e-4
    )
    assert problem["data:handling_qualities:lateral:derivatives:CY:rollrate"] == pytest.approx(
        cY_p, abs=1e-4
    )
    assert problem["data:handling_qualities:lateral:derivatives:Cl:rollrate"] == pytest.approx(
        cl_p, abs=1e-4
    )
    assert problem["data:handling_qualities:lateral:derivatives:Cn:rollrate"] == pytest.approx(
        cn_p, abs=1e-4
    )
    assert problem["data:handling_qualities:lateral:derivatives:CY:yawrate"] == pytest.approx(
        cY_r, abs=1e-4
    )
    assert problem["data:handling_qualities:lateral:derivatives:Cl:yawrate"] == pytest.approx(
        cl_r, abs=1e-4
    )
    assert problem["data:handling_qualities:lateral:derivatives:Cn:yawrate"] == pytest.approx(
        cn_r, abs=1e-4
    )


