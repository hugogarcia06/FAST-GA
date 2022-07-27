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
import math
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from importlib.resources import path

from .. import resources as local_resources

from platform import system
from pathlib import Path
from tempfile import TemporaryDirectory

from openmdao.utils.file_wrap import InputFileGenerator

from fastga.models.handling_qualities.stability_derivatives.external.openvsp.compute_stab import ComputeSTABopenvsp
from tests.testing_utilities import run_system, get_indep_var_comp, list_inputs
from tests.xfoil_exe.get_xfoil import get_xfoil_path

from fastga.models.aerodynamics.external.xfoil import resources
from fastga.models.handling_qualities.aircraft_modes.aircraft_modes_computation import AircraftModesComputation
from ..check_modes.aircraft_modes_analysis import AircraftModesAnalysis
from ..check_modes.lateral_directional.check_dutch_roll import CheckDutchRoll
from ..check_modes.longitudinal.check_short_period import CheckShortPeriod
from ..stability_derivatives.stability_derivatives import StabilityDerivatives

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


def stability(
    add_fuselage: bool,
    use_openvsp: bool,
    XML_FILE: str,
    ):
    """Compute stability coefficients for the whole aircraft using OpenVSP or semi-empirical methods"""
    # Create result temporary directory
    results_folder = _create_tmp_directory()

    # Transfer saved polar results to temporary folder
    tmp_folder = polar_result_transfer()

    # Research independent input value in .xml file
    # noinspection PyTypeChecker
    ivc = get_indep_var_comp(
        list_inputs(StabilityDerivatives(
            airplane_file=XML_FILE,
            result_folder_path=results_folder.name,
            add_fuselage=add_fuselage,
            use_openvsp=use_openvsp
        )), __file__, XML_FILE
    )

    # Run problem twice
    start = time.time()
    # noinspection PyTypeChecker
    problem = run_system(
        StabilityDerivatives(
            airplane_file=XML_FILE,
            result_folder_path=results_folder.name,
            add_fuselage=add_fuselage,
            use_openvsp=use_openvsp
        ),
        ivc,
    )

    stop = time.time()
    duration_1st_run = stop - start
    start = time.time()
    # noinspection PyTypeChecker
    run_system(
        StabilityDerivatives(
            airplane_file=XML_FILE,
            result_folder_path=results_folder.name,
            add_fuselage=add_fuselage,
            use_openvsp=use_openvsp
        ),
        ivc,
    )
    stop = time.time()

    CL_alpha = problem.get_val("data:handling_qualities:longitudinal:derivatives:CL:alpha", units="rad**-1")
    CD_alpha = problem.get_val("data:handling_qualities:longitudinal:derivatives:CD:alpha", units="rad**-1")
    Cm_alpha = problem.get_val("data:handling_qualities:longitudinal:derivatives:Cm:alpha", units="rad**-1")
    CL_speed = problem.get_val("data:handling_qualities:longitudinal:derivatives:CL:speed", units="rad**-1")
    CD_speed = problem.get_val("data:handling_qualities:longitudinal:derivatives:CD:speed", units="rad**-1")
    Cm_speed = problem.get_val("data:handling_qualities:longitudinal:derivatives:Cm:speed", units="rad**-1")
    CL_pitchrate = problem.get_val("data:handling_qualities:longitudinal:derivatives:CL:pitchrate", units="rad**-1")
    CD_pitchrate = problem.get_val("data:handling_qualities:longitudinal:derivatives:CD:pitchrate", units="rad**-1")
    Cm_pitchrate = problem.get_val("data:handling_qualities:longitudinal:derivatives:Cm:pitchrate", units="rad**-1")

    CY_beta = problem.get_val("data:handling_qualities:lateral:derivatives:CY:beta", units="rad**-1")
    Cl_beta = problem.get_val("data:handling_qualities:lateral:derivatives:Cl:beta", units="rad**-1")
    Cn_beta = problem.get_val("data:handling_qualities:lateral:derivatives:Cn:beta", units="rad**-1")
    CY_rollrate = problem.get_val("data:handling_qualities:lateral:derivatives:CY:rollrate", units="rad**-1")
    Cl_rollrate = problem.get_val("data:handling_qualities:lateral:derivatives:Cl:rollrate", units="rad**-1")
    Cn_rollrate = problem.get_val("data:handling_qualities:lateral:derivatives:Cn:rollrate", units="rad**-1")
    CY_yawrate = problem.get_val("data:handling_qualities:lateral:derivatives:CY:yawrate", units="rad**-1")
    Cl_yawrate = problem.get_val("data:handling_qualities:lateral:derivatives:Cl:yawrate", units="rad**-1")
    Cn_yawrate = problem.get_val("data:handling_qualities:lateral:derivatives:Cn:yawrate", units="rad**-1")

    # Write the results in a file
    parser = InputFileGenerator()
    RESULTS_FILE = "results_DATCOM"
    if use_openvsp:
        RESULTS_FILE = "results_openvsp"

    with path(local_resources, "template_results.txt") as input_template_path:
        parser.set_template_file(str(input_template_path))
        parser.set_generated_file(RESULTS_FILE)
        parser.mark_anchor("CL_alph")
        parser.transfer_var(round(float(CL_alpha), 5), 0, 2)
        parser.mark_anchor("CD_alph")
        parser.transfer_var(round(float(CD_alpha), 5), 0, 2)
        parser.mark_anchor("Cm_alph")
        parser.transfer_var(round(float(Cm_alpha), 5), 0, 2)
        parser.reset_anchor()
        parser.mark_anchor("CL_u")
        parser.transfer_var(round(float(CL_speed), 5), 0, 8)
        parser.mark_anchor("CD_u")
        parser.transfer_var(round(float(CD_speed), 5), 0, 8)
        parser.mark_anchor("Cm_u")
        parser.transfer_var(round(float(Cm_speed), 5), 0, 8)
        parser.reset_anchor()
        CL_q = round(float(CL_pitchrate), 5)
        parser.mark_anchor("CL_qqqq")
        parser.transfer_var(CL_q, 0, 5)
        parser.mark_anchor("CD_qqqq")
        parser.transfer_var(round(float(CD_pitchrate), 5), 0, 5)
        parser.mark_anchor("Cm_qqqq")
        parser.transfer_var(round(float(Cm_pitchrate), 5), 0, 5)
        parser.reset_anchor()
        parser.mark_anchor("CY_beta")
        parser.transfer_var(round(float(CY_beta), 5), 0, 3)
        parser.mark_anchor("Cl_beta")
        parser.transfer_var(round(float(Cl_beta), 5), 0, 3)
        parser.mark_anchor("Cn_beta")
        parser.transfer_var(round(float(Cn_beta), 5), 0, 3)
        parser.reset_anchor()
        parser.mark_anchor("CY_pppp")
        parser.transfer_var(round(float(CY_rollrate), 5), 0, 4)
        parser.mark_anchor("Cl_pppp")
        parser.transfer_var(round(float(Cl_rollrate), 5), 0, 4)
        parser.mark_anchor("Cn_pppp")
        parser.transfer_var(round(float(Cn_rollrate), 5), 0, 4)
        parser.reset_anchor()
        parser.mark_anchor("CY_rrrr")
        parser.transfer_var(round(float(CY_yawrate), 5), 0, 6)
        parser.mark_anchor("Cl_rrrr")
        parser.transfer_var(round(float(Cl_yawrate), 5), 0, 6)
        parser.mark_anchor("Cn_rrrr")
        parser.transfer_var(round(float(Cn_yawrate), 5), 0, 6)
        parser.reset_anchor()
        parser.generate()

    duration_2nd_run = stop - start

    # Retrieve polar results from temporary folder
    polar_result_retrieve(tmp_folder)

    # Remove existing result files
    results_folder.cleanup()

    # Check obtained value(s) is/(are) correct
    assert (duration_2nd_run / duration_1st_run) <= 0.1

    # Return problem for complementary values check
    return problem


def alpha_derivatives(use_openvsp, add_fuselage, XML_FILE):
    # Run problem and check obtained value(s) is/(are) correct

    # Create result temporary directory
    results_folder = _create_tmp_directory()

    # Transfer saved polar results to temporary folder
    tmp_folder = polar_result_transfer()

    ivc = get_indep_var_comp(
        list_inputs(StabilityDerivatives(
            result_folder_path=results_folder.name,
            add_fuselage=add_fuselage,
            use_openvsp=use_openvsp
        )), __file__, XML_FILE
    )
    problem = run_system(
        StabilityDerivatives(
            result_folder_path=results_folder.name,
            add_fuselage=add_fuselage,
            use_openvsp=use_openvsp
        ), ivc
    )

    # Retrieve polar results from temporary folder
    polar_result_retrieve(tmp_folder)

    # Remove existing result files
    results_folder.cleanup()


    CL_alpha = problem.get_val("data:handling_qualities:longitudinal:derivatives:CL:alpha", units="rad**-1")

    return problem


def aircraft_modes(add_fuselage, use_openvsp, XML_FILE):

    # Create result temporary directory
    results_folder = _create_tmp_directory()

    # Transfer saved polar results to temporary folder
    tmp_folder = polar_result_transfer()

    ivc = get_indep_var_comp(
        list_inputs(AircraftModesComputation(
            airplane_file=XML_FILE,
            plot_modes=True,
            result_folder_path=results_folder.name,
            add_fuselage=add_fuselage,
            use_openvsp=use_openvsp
        )), __file__, XML_FILE
    )

    problem = run_system(
        AircraftModesComputation(
            airplane_file=XML_FILE,
            plot_modes=True,
            result_folder_path=results_folder.name,
            add_fuselage=add_fuselage,
            use_openvsp=use_openvsp
        ), ivc
    )

    # Retrieve polar results from temporary folder
    polar_result_retrieve(tmp_folder)

    # Copy temporary folder into Stage DCAS / Results
    original = results_folder.name
    target = "C:/Users/hugog/OneDrive - Universidad Politécnica de Madrid/GoodNotes 5/Máster Ingeniería Aeroespacial (MUIA)/2º MUIA (SUPAERO)/2º Sem/Stage DCAS/RESULTS/Aircraft Modes"
    if XML_FILE == "beechcraft_76.xml":
        target = "C:/Users/hugog/OneDrive - Universidad Politécnica de Madrid/GoodNotes 5/Máster Ingeniería Aeroespacial (MUIA)/2º MUIA (SUPAERO)/2º Sem/Stage DCAS/RESULTS/Beechcraft 76/Aircraft Modes"
    elif XML_FILE == "cirrus_sr22.xml":
        target = "C:/Users/hugog/OneDrive - Universidad Politécnica de Madrid/GoodNotes 5/Máster Ingeniería Aeroespacial (MUIA)/2º MUIA (SUPAERO)/2º Sem/Stage DCAS/RESULTS/Cirrus SR22/Aircraft Modes"
    elif XML_FILE == "cessna_182.xml":
        target = "C:/Users/hugog/OneDrive - Universidad Politécnica de Madrid/GoodNotes 5/Máster Ingeniería Aeroespacial (MUIA)/2º MUIA (SUPAERO)/2º Sem/Stage DCAS/RESULTS/Cessna 182/Aircraft Modes"

    shutil.move(original, target)

    # Remove existing result files
    results_folder.cleanup()

    damp_ph = problem.get_val("data:handling_qualities:longitudinal:modes:phugoid:damping_ratio")
    wn_ph = problem.get_val("data:handling_qualities:longitudinal:modes:phugoid:undamped_frequency", units="s**-1")


def check_aircraft_modes(add_fuselage, use_openvsp, XML_FILE):
    # Create result temporary directory
    results_folder = _create_tmp_directory()

    # Transfer saved polar results to temporary folder
    tmp_folder = polar_result_transfer()

    ivc = get_indep_var_comp(
        list_inputs(AircraftModesAnalysis(
            airplane_file=XML_FILE,
            result_folder_path=results_folder.name,
            add_fuselage=add_fuselage,
            use_openvsp=use_openvsp
        )), __file__, XML_FILE
    )

    problem = run_system(
        AircraftModesAnalysis(
            airplane_file=XML_FILE,
            result_folder_path=results_folder.name,
            add_fuselage=add_fuselage,
            use_openvsp=use_openvsp
        ), ivc
    )

    # Retrieve polar results from temporary folder
    polar_result_retrieve(tmp_folder)

    # Copy temporary folder into Stage DCAS / Results
    original = results_folder.name
    target = "C:/Users/hugog/OneDrive - Universidad Politécnica de Madrid/GoodNotes 5/Máster Ingeniería Aeroespacial (MUIA)/2º MUIA (SUPAERO)/2º Sem/Stage DCAS/RESULTS/Check Modes"
    if XML_FILE == "beechcraft_76.xml":
        target = "C:/Users/hugog/OneDrive - Universidad Politécnica de Madrid/GoodNotes 5/Máster Ingeniería Aeroespacial (MUIA)/2º MUIA (SUPAERO)/2º Sem/Stage DCAS/RESULTS/Beechcraft 76/Check Modes"
    elif XML_FILE == "cirrus_sr22.xml":
        target = "C:/Users/hugog/OneDrive - Universidad Politécnica de Madrid/GoodNotes 5/Máster Ingeniería Aeroespacial (MUIA)/2º MUIA (SUPAERO)/2º Sem/Stage DCAS/RESULTS/Cirrus SR22/Check Modes"
    elif XML_FILE == "cessna_182.xml":
        target = "C:/Users/hugog/OneDrive - Universidad Politécnica de Madrid/GoodNotes 5/Máster Ingeniería Aeroespacial (MUIA)/2º MUIA (SUPAERO)/2º Sem/Stage DCAS/RESULTS/Cessna 182/Check Modes"

    shutil.move(original, target)

    # Remove existing result files
    results_folder.cleanup()

    damp_ph = problem.get_val("data:handling_qualities:longitudinal:modes:phugoid:damping_ratio")
    wn_ph = problem.get_val("data:handling_qualities:longitudinal:modes:phugoid:undamped_frequency", units="s**-1")


def check_aircraft_modes_modified(add_fuselage, use_openvsp, XML_FILE, ivc):
    # Create result temporary directory
    results_folder = _create_tmp_directory()

    # Transfer saved polar results to temporary folder
    tmp_folder = polar_result_transfer()

    problem = run_system(
        AircraftModesAnalysis(
            result_folder_path=results_folder.name,
            add_fuselage=add_fuselage,
            use_openvsp=use_openvsp
        ), ivc
    )

    # Retrieve polar results from temporary folder
    polar_result_retrieve(tmp_folder)

    # Copy temporary folder into Stage DCAS / Results
    original = results_folder.name
    target = "C:/Users/hugog/OneDrive - Universidad Politécnica de Madrid/GoodNotes 5/Máster Ingeniería Aeroespacial (MUIA)/2º MUIA (SUPAERO)/2º Sem/Stage DCAS/RESULTS"
    if XML_FILE == "beechcraft_76.xml":
        target = "C:/Users/hugog/OneDrive - Universidad Politécnica de Madrid/GoodNotes 5/Máster Ingeniería Aeroespacial (MUIA)/2º MUIA (SUPAERO)/2º Sem/Stage DCAS/RESULTS/Beechcraft 76"
    elif XML_FILE == "cirrus_sr22.xml":
        target = "C:/Users/hugog/OneDrive - Universidad Politécnica de Madrid/GoodNotes 5/Máster Ingeniería Aeroespacial (MUIA)/2º MUIA (SUPAERO)/2º Sem/Stage DCAS/RESULTS/Cirrus SR22"
    elif XML_FILE == "cessna_182.xml":
        target = "C:/Users/hugog/OneDrive - Universidad Politécnica de Madrid/GoodNotes 5/Máster Ingeniería Aeroespacial (MUIA)/2º MUIA (SUPAERO)/2º Sem/Stage DCAS/RESULTS/Cessna 182"

    shutil.move(original, target)

    # Remove existing result files
    results_folder.cleanup()

    damp_ph = problem.get_val("data:handling_qualities:longitudinal:modes:phugoid:damping_ratio")
    wn_ph = problem.get_val("data:handling_qualities:longitudinal:modes:phugoid:undamped_frequency", units="s**-1")


def dynamic_tail_sizing_loop(add_fuselage, use_openvsp, XML_FILE):
    # Create result temporary directory
    results_folder = _create_tmp_directory()

    ht_area_original = 2.67406   # in square meters
    ht_areas = [ht_area_original*0.75, ht_area_original*0.9, ht_area_original, ht_area_original*1.1, ht_area_original*1.25, ht_area_original*1.5]

    sp_real_parts = []
    sp_imag_parts = []
    x_cg = 0.0
    for ht_area in ht_areas:

        ivc = get_indep_var_comp(
            list_inputs(AircraftModesComputation(
                result_folder_path=results_folder.name,
                add_fuselage=add_fuselage,
                use_openvsp=use_openvsp
            )), __file__, XML_FILE
        )
        ivc.add_output("data:geometry:horizontal_tail:area", val=ht_area, units="m**2")

        problem = run_system(
            AircraftModesComputation(
                result_folder_path=results_folder.name,
                add_fuselage=add_fuselage,
                use_openvsp=use_openvsp
            ), ivc
        )

        x_cg = problem.get_val("data:reference_flight_condition:CG:x")
        real_sp = problem.get_val("data:handling_qualities:longitudinal:modes:short_period:real_part")
        imag_sp = problem.get_val("data:handling_qualities:longitudinal:modes:short_period:imag_part")

        sp_real_parts.append(real_sp)
        sp_imag_parts.append(imag_sp)


    ### PLOT ###
    fig, ax = plt.subplots()
    ax.set_title("Short Period Eigenvalues for different HT Areas for $X_{CG} = %s $" % float(x_cg))
    ax.set_xlabel(r"$n$")
    ax.set_ylabel(r"$jw$")
    ax.grid(visible=True, which="both")
    # x-axis
    ax.plot(np.linspace(-10, 10, 1000), 0.0 * np.linspace(-10, 10, 1000), color="black")
    # y-axis
    ax.plot(0.0 * np.linspace(-10, 10, 1000), np.linspace(-10, 10, 1000), color="black")

    for i in range(len(ht_areas)):

        label = r"$S_{HTP} = $" + str(ht_areas[i])
        ax.scatter(sp_real_parts[i], sp_imag_parts[i], label=label)

    ax.legend(loc="upper right")
    min_real_parts = min(sp_real_parts)
    max_real_parts = max(sp_real_parts)
    max_imag_parts = max(sp_imag_parts)
    ax.set_xbound(min_real_parts * 1.25, max(abs(max_real_parts), 1.0) * 1.5)
    ax.set_ybound(-0.1, max_imag_parts * 1.5)
    plt.show()

    results_dir = results_folder.name
    plot_name = "sp_eigenvalues_variation.png"

    if not os.path.isdir(results_dir):
        os.makedirs(results_dir)

    fig.savefig(results_dir + plot_name)

    # Copy temporary folder into Stage DCAS / Results
    original = results_folder.name
    target = "C:/Users/hugog/OneDrive - Universidad Politécnica de Madrid/GoodNotes 5/Máster Ingeniería Aeroespacial (MUIA)/2º MUIA (SUPAERO)/2º Sem/Stage DCAS/RESULTS/Dynamic Tail Sizing"

    shutil.move(original, target)

    # Remove existing result files
    results_folder.cleanup()


def short_period_tail_sizing(add_fuselage, use_openvsp, XML_FILE, XML_OPTIM_FILE):
    def get_damping_limits(damping_ratio):
        if damping_ratio == 0.0:
            x = np.linspace(-100, 0.0, 100) * 0.0
            y = np.linspace(-100, 0.0, 100)
        elif damping_ratio == 1.0 or damping_ratio > 1.0:
            x = np.linspace(-100, 0.0, 100)
            y = np.linspace(-100, 0.0, 100) * 0.0
        else:
            phi = math.asin(damping_ratio)
            phi = math.pi / 2 - phi
            x = np.linspace(-100, 0.0, 100)
            y = math.tan(-phi) * x

        return x, y

    def get_frequency_limits(wn):
        angle = np.linspace(math.pi, math.pi / 2.0, 100)

        x = wn * np.cos(angle)
        y = wn * np.sin(angle)

        return x, y

    # Create result temporary directory
    results_folder = _create_tmp_directory()

    # First we give default reference flight conditions
    ref_flight_condition_dict = {}

    ivc = get_indep_var_comp(
        list_inputs(AircraftModesComputation(
            airplane_file=XML_FILE,
            reference_flight_condition=ref_flight_condition_dict,
            result_folder_path=results_folder.name,
            add_fuselage=add_fuselage,
            use_openvsp=use_openvsp
        )), __file__, XML_FILE
    )

    problem = run_system(
        AircraftModesComputation(
            airplane_file=XML_FILE,
            reference_flight_condition=ref_flight_condition_dict,
            result_folder_path=results_folder.name,
            add_fuselage=add_fuselage,
            use_openvsp=use_openvsp
        ), ivc
    )

    cg_aft = problem.get_val("data:weight:aircraft:CG:aft:x")
    cg_fwd = problem.get_val("data:weight:aircraft:CG:fwd:x")

    # cg_x_positions = np.linspace(cg_fwd, cg_aft, 5)
    cg_x_positions = np.linspace(3.474, 3.474, 1)
    # ht_areas = np.linspace(2.0, 6.0, 6)
    ht_areas = np.linspace(2.5, 3.7, 10)

    sp_real_parts = []
    sp_imag_parts = []

    for cg_x in cg_x_positions:

        ref_flight_condition_dict = {
            "mach": 0.201,
            "altitude": 5000.0,
            "theta": 0.0,
            "weight": 1450.0,
            "cg_x": cg_x,
        }

        area_real_parts = []
        area_imag_parts = []

        for ht_area in ht_areas:

            ivc = get_indep_var_comp(
                list_inputs(AircraftModesComputation(
                    airplane_file=XML_OPTIM_FILE,
                    result_folder_path=results_folder.name,
                    reference_flight_condition=ref_flight_condition_dict,
                    add_fuselage=add_fuselage,
                    use_openvsp=use_openvsp
                )), __file__, XML_OPTIM_FILE
            )
            ivc.add_output("data:geometry:horizontal_tail:area", val=ht_area, units="m**2")

            problem = run_system(
                AircraftModesComputation(
                    airplane_file=XML_OPTIM_FILE,
                    result_folder_path=results_folder.name,
                    reference_flight_condition=ref_flight_condition_dict,
                    add_fuselage=add_fuselage,
                    use_openvsp=use_openvsp
                ), ivc
            )

            real_sp = problem.get_val("data:handling_qualities:longitudinal:modes:short_period:real_part")
            imag_sp = problem.get_val("data:handling_qualities:longitudinal:modes:short_period:imag_part")

            area_real_parts.append(real_sp)
            area_imag_parts.append(imag_sp)

        sp_real_parts.append(area_real_parts)
        sp_imag_parts.append(area_imag_parts)


    flight_phase_category = problem.get_val("data:reference_flight_condition:flight_phase_category")
    q = problem.get_val("data:reference_flight_condition:dynamic_pressure")
    CL_alpha = problem.get_val("data:handling_qualities:longitudinal:derivatives:CL:alpha")
    S = problem.get_val("data:geometry:wing:area")
    W = problem.get_val("data:reference_flight_condition:weight")
    n_alpha = q * CL_alpha * S / W

    z_sp_reqs, wn_sp_reqs = CheckShortPeriod.get_short_period_requirements(flight_phase_category, n_alpha)
    z_sp_min_req_1 = z_sp_reqs[0]
    z_sp_max_req_1 = z_sp_reqs[1]
    wn_sp_min_req_1 = wn_sp_reqs[0]
    wn_sp_max_req_1 = wn_sp_reqs[1]

    ### PLOT ###
    fig, ax = plt.subplots(figsize=(11.2, 8.4))
    title = "Short Period Eigenvalues for different HT Areas and CG positions, " \
            + "altitude = " + str(ref_flight_condition_dict["altitude"]) + " ft. " \
            + "q = " + str(round(float(q), 3)) + " Pa. "

    ax.set_title(title)
    # ax.set_title("Short Period Eigenvalues for different HT Areas for $X_{CG} = %s $" % float(cg_x))
    ax.set_xlabel(r"$n$")
    ax.set_ylabel(r"$jw$")
    ax.grid(visible=True, which="both")
    # x-axis
    ax.plot(np.linspace(-10, 10, 1000), 0.0 * np.linspace(-10, 10, 1000), color="black")
    # y-axis
    ax.plot(0.0 * np.linspace(-10, 10, 1000), np.linspace(-10, 10, 1000), color="black")

    markers = [".", "+", "x", "*", "d"]
    # colors = ["b", "g", "r", "c", "m", "y", "k", "orange"]
    colors = cm.rainbow(np.linspace(0, 1, len(ht_areas)))
    for i in range(len(cg_x_positions)):
        for j in range(len(ht_areas)):

            if j == 0:
                label = r"$S_{HTP} = $" + str(round(ht_areas[j], 3)) + " " + r"$x_{CG} = $" + str(round(float(cg_x_positions[i]), 3))
            else:
                label = r"$S_{HTP} = $" + str(round(ht_areas[j], 3))

            marker = markers[i]
            color = colors[j]
            ax.scatter(sp_real_parts[i][j], sp_imag_parts[i][j], label=label, marker=marker, color=color)

    # Damping ratio limits
    # Level 1
    x_damp_min_1, y_damp_min_1 = get_damping_limits(z_sp_min_req_1)
    ax.plot(x_damp_min_1, y_damp_min_1, linestyle="dashdot", color="red", label="Level 1")
    x_damp_max_1, y_damp_max_1 = get_damping_limits(z_sp_max_req_1)
    ax.plot(x_damp_max_1, y_damp_max_1, linestyle="dashdot", color="red", label="")

    # Frequency limits
    # Level 1
    x_freq_min_1, y_freq_min_1 = get_frequency_limits(wn_sp_min_req_1)
    ax.plot(x_freq_min_1, y_freq_min_1, linestyle="--", color="red", label="")
    x_freq_max_1, y_freq_max_1 = get_frequency_limits(wn_sp_max_req_1)
    ax.plot(x_freq_max_1, y_freq_max_1, linestyle="--", color="red", label="")

    ax.legend(loc="upper right")
    min_real_parts = min(min(sp_real_parts))
    max_real_parts = max(max(sp_real_parts))
    max_imag_parts = max(max(sp_imag_parts))
    ax.set_xbound(min_real_parts * 1.25, 1.0)
    ax.set_ybound(-1.5, max_imag_parts * 1.5)
    plt.show()

    results_dir = results_folder.name
    plot_name = "sp_eigenvalues_variation.png"
    fig_dir = os.path.join(results_dir, plot_name)

    fig.savefig(fig_dir)

    # Copy temporary folder into Stage DCAS / Results
    original = results_folder.name
    target = "C:/Users/hugog/OneDrive - Universidad Politécnica de Madrid/GoodNotes 5/Máster Ingeniería Aeroespacial (MUIA)/2º MUIA (SUPAERO)/2º Sem/Stage DCAS/RESULTS/Short Period Tail Sizing"
    if XML_FILE == "beechcraft_76.xml":
        target = "C:/Users/hugog/OneDrive - Universidad Politécnica de Madrid/GoodNotes 5/Máster Ingeniería Aeroespacial (MUIA)/2º MUIA (SUPAERO)/2º Sem/Stage DCAS/RESULTS/Beechcraft 76/Short Period Tail Sizing"
    elif XML_FILE == "cirrus_sr22.xml":
        target = "C:/Users/hugog/OneDrive - Universidad Politécnica de Madrid/GoodNotes 5/Máster Ingeniería Aeroespacial (MUIA)/2º MUIA (SUPAERO)/2º Sem/Stage DCAS/RESULTS/Cirrus SR22/Short Period Tail Sizing"
    elif XML_FILE == "cessna_182.xml":
        target = "C:/Users/hugog/OneDrive - Universidad Politécnica de Madrid/GoodNotes 5/Máster Ingeniería Aeroespacial (MUIA)/2º MUIA (SUPAERO)/2º Sem/Stage DCAS/RESULTS/Cessna 182/Short Period Tail Sizing"

    shutil.move(original, target)

    # Remove existing result files
    results_folder.cleanup()



def dutch_roll_tail_sizing(add_fuselage, use_openvsp, XML_FILE, XML_OPTIM_FILE):
    def get_damping_limits(damping_ratio):
        if damping_ratio == 0.0:
            x = np.linspace(-100, 0.0, 100) * 0.0
            y = np.linspace(-100, 0.0, 100)
        elif damping_ratio == 1.0 or damping_ratio > 1.0:
            x = np.linspace(-100, 0.0, 100)
            y = np.linspace(-100, 0.0, 100) * 0.0
        else:
            phi = math.asin(damping_ratio)
            phi = math.pi / 2 - phi
            x = np.linspace(-100, 0.0, 100)
            y = math.tan(-phi) * x

        return x, y

    def get_vertical_limits(z_wn_product):
        x = -z_wn_product * np.ones(1000)
        y = np.linspace(-100, 100, 1000)

        return x, y

    def get_frequency_limits(wn):
        angle = np.linspace(math.pi, math.pi / 2.0, 100)

        x = wn * np.cos(angle)
        y = wn * np.sin(angle)

        return x, y

    # Create result temporary directory
    results_folder = _create_tmp_directory()

    # First we give default reference flight conditions
    ref_flight_condition_dict = {}

    ivc = get_indep_var_comp(
        list_inputs(AircraftModesComputation(
            airplane_file=XML_FILE,
            reference_flight_condition=ref_flight_condition_dict,
            result_folder_path=results_folder.name,
            add_fuselage=add_fuselage,
            use_openvsp=use_openvsp
        )), __file__, XML_FILE
    )

    problem = run_system(
        AircraftModesComputation(
            airplane_file=XML_FILE,
            reference_flight_condition=ref_flight_condition_dict,
            result_folder_path=results_folder.name,
            add_fuselage=add_fuselage,
            use_openvsp=use_openvsp
        ), ivc
    )

    cg_aft = problem.get_val("data:weight:aircraft:CG:aft:x")
    cg_fwd = problem.get_val("data:weight:aircraft:CG:fwd:x")
    MTOW = problem.get_val("data:weight:aircraft:MTOW")
    cg_x = (cg_aft + cg_fwd) / 2.0

    dihedral_angles = np.linspace(0.0, 10.0, 5)
    vt_areas = np.linspace(1.0, 3.0, 6)

    sp_real_parts = []
    sp_imag_parts = []

    for dihedral in dihedral_angles:

        ref_flight_condition_dict = {
            "mach": 0.201,
            "altitude": 5000.0,
            "theta": 0.0,
            "weight": MTOW * 0.9,
            "cg_x": cg_x,
        }

        area_real_parts = []
        area_imag_parts = []

        for vt_area in vt_areas:
            ivc = get_indep_var_comp(
                list_inputs(AircraftModesComputation(
                    airplane_file=XML_OPTIM_FILE,
                    result_folder_path=results_folder.name,
                    reference_flight_condition=ref_flight_condition_dict,
                    add_fuselage=add_fuselage,
                    use_openvsp=use_openvsp
                )), __file__, XML_OPTIM_FILE
            )
            ivc.add_output("data:geometry:vertical_tail:area", val=vt_area, units="m**2")
            ivc.add_output("data:geometry:wing:dihedral", val=dihedral, units="deg")

            problem = run_system(
                AircraftModesComputation(
                    airplane_file=XML_OPTIM_FILE,
                    result_folder_path=results_folder.name,
                    reference_flight_condition=ref_flight_condition_dict,
                    add_fuselage=add_fuselage,
                    use_openvsp=use_openvsp
                ), ivc
            )

            real_sp = problem.get_val("data:handling_qualities:longitudinal:modes:short_period:real_part")
            imag_sp = problem.get_val("data:handling_qualities:longitudinal:modes:short_period:imag_part")

            area_real_parts.append(real_sp)
            area_imag_parts.append(imag_sp)

        sp_real_parts.append(area_real_parts)
        sp_imag_parts.append(area_imag_parts)

    flight_phase_category = problem.get_val("data:reference_flight_condition:flight_phase_category")
    # aircraft_class = problem.get_val("data:geometry:aircraft:class")
    aircraft_class = 1.0
    q = problem.get_val("data:reference_flight_condition:dynamic_pressure")


    level_1_dr_reqs, level_2_dr_reqs, level_3_dr_reqs = CheckDutchRoll.get_dutch_roll_requirements(
        aircraft_class, flight_phase_category)
    min_z_dr_1 = level_1_dr_reqs[0]
    min_z_wn_dr_1 = level_1_dr_reqs[1]
    min_wn_dr_1 = level_1_dr_reqs[2]

    ### PLOT ###
    # fig, ax = plt.subplots(figsize=(11.2, 8.4))
    fig, ax = plt.subplots()
    title = "Dutch Roll Eigenvalues for different VT Areas and Dihedral Angles, " \
            + "altitude = " + str(ref_flight_condition_dict["altitude"]) + " ft. " \
            + "q = " + str(round(float(q), 3)) + " Pa. "

    ax.set_title(title)
    ax.set_xlabel(r"$n$")
    ax.set_ylabel(r"$jw$")
    ax.grid(visible=True, which="both")
    # x-axis
    ax.plot(np.linspace(-10, 10, 1000), 0.0 * np.linspace(-10, 10, 1000), color="black")
    # y-axis
    ax.plot(0.0 * np.linspace(-10, 10, 1000), np.linspace(-10, 10, 1000), color="black")

    markers = [".", "+", "x", "*", "d"]
    colors = ["b", "g", "r", "c", "m", "y", "k", "orange"]
    for i in range(len(dihedral_angles)):
        for j in range(len(vt_areas)):

            if j == 0:
                label = r"$S_{VTP} = $" + str(vt_areas[j]) + " " + r"$\Gamma = $" + str(
                    round(float(dihedral_angles[i]), 3))
            else:
                label = r"$S_{VTP} = $" + str(vt_areas[j])

            marker = markers[i]
            color = colors[j]
            ax.scatter(sp_real_parts[i][j], sp_imag_parts[i][j], label=label, marker=marker, color=color)

    # Limits
    # Damping ratio limits
    x_1, y_1 = get_damping_limits(min_z_dr_1)
    ax.plot(x_1, y_1, linestyle="dashdot", color="red")

    # Frequency limits

    x_1, y_1 = get_frequency_limits(min_wn_dr_1)
    ax.plot(x_1, y_1, linestyle="--", color="red")

    # Damping-frequency product limit
    x_1, y_1 = get_vertical_limits(min_z_wn_dr_1)
    ax.plot(x_1, y_1, linestyle="--", color="red")

    ax.legend(loc="upper right")
    min_real_parts = min(min(sp_real_parts))
    max_real_parts = max(max(sp_real_parts))
    max_imag_parts = max(max(sp_imag_parts))
    ax.set_xbound(min_real_parts * 1.25, max(abs(max_real_parts), 1.0) * 1.5)
    ax.set_ybound(-0.1, max_imag_parts * 1.5)
    plt.show()

    results_dir = results_folder.name
    plot_name = "dr_eigenvalues_variation.png"
    fig_dir = os.path.join(results_dir, plot_name)

    fig.savefig(fig_dir)

    # Copy temporary folder into Stage DCAS / Results
    original = results_folder.name
    target = "C:/Users/hugog/OneDrive - Universidad Politécnica de Madrid/GoodNotes 5/Máster Ingeniería Aeroespacial (MUIA)/2º MUIA (SUPAERO)/2º Sem/Stage DCAS/RESULTS/Dutch Roll Tail Sizing"
    if XML_FILE == "beechcraft_76.xml":
        target = "C:/Users/hugog/OneDrive - Universidad Politécnica de Madrid/GoodNotes 5/Máster Ingeniería Aeroespacial (MUIA)/2º MUIA (SUPAERO)/2º Sem/Stage DCAS/RESULTS/Beechcraft 76/Dutch Roll Tail Sizing"
    elif XML_FILE == "cirrus_sr22.xml":
        target = "C:/Users/hugog/OneDrive - Universidad Politécnica de Madrid/GoodNotes 5/Máster Ingeniería Aeroespacial (MUIA)/2º MUIA (SUPAERO)/2º Sem/Stage DCAS/RESULTS/Cirrus SR22/Dutch Roll Tail Sizing"
    elif XML_FILE == "cessna_182.xml":
        target = "C:/Users/hugog/OneDrive - Universidad Politécnica de Madrid/GoodNotes 5/Máster Ingeniería Aeroespacial (MUIA)/2º MUIA (SUPAERO)/2º Sem/Stage DCAS/RESULTS/Cessna 182/Dutch Roll Tail Sizing"

    shutil.move(original, target)

    # Remove existing result files
    results_folder.cleanup()



