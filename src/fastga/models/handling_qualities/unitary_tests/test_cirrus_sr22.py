#  This file is part of FAST : A framework for rapid Overall Aircraft Design
#  Copyright (C) 2020  ONERA & ISAE-SUPAERO
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

from .test_functions import check_aircraft_modes, aircraft_modes, short_period_tail_sizing, dutch_roll_tail_sizing
from ..compute_static_margin import ComputeStaticMargin
from ..tail_sizing.update_vt_area import UpdateVTArea
from ..tail_sizing.update_ht_area import UpdateHTArea
from ..tail_sizing.compute_to_rotation_limit import ComputeTORotationLimitGroup
from ..tail_sizing.compute_balked_landing_limit import ComputeBalkedLandingLimit

from tests.testing_utilities import run_system, get_indep_var_comp, list_inputs

from .dummy_engines import ENGINE_WRAPPER_SR22 as ENGINE_WRAPPER

XML_FILE = "cirrus_sr22.xml"


def test_update_vt_area():
    """Tests computation of the vertical tail area"""

    # Research independent input value in .xml file
    input_vars = get_indep_var_comp(
        list_inputs(UpdateVTArea(propulsion_id=ENGINE_WRAPPER)), __file__, XML_FILE
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(UpdateVTArea(propulsion_id=ENGINE_WRAPPER), input_vars)
    vt_area = problem.get_val("data:geometry:vertical_tail:area", units="m**2")
    assert vt_area == pytest.approx(1.41, abs=1e-2)

    vt_area_constraints_cruise = problem.get_val(
        "data:constraints:vertical_tail:target_cruise_stability", units="m**2"
    )
    assert vt_area_constraints_cruise == pytest.approx(0.0, abs=1e-2)
    vt_area_constraints_crosswind = problem.get_val(
        "data:constraints:vertical_tail:crosswind_landing", units="m**2"
    )
    assert vt_area_constraints_crosswind == pytest.approx(0.0, abs=1e-2)
    vt_area_constraints_eo_climb = problem.get_val(
        "data:constraints:vertical_tail:engine_out_climb", units="m**2"
    )
    assert vt_area_constraints_eo_climb == pytest.approx(1.41, abs=1e-2)
    vt_area_constraints_eo_takeoff = problem.get_val(
        "data:constraints:vertical_tail:engine_out_takeoff", units="m**2"
    )
    assert vt_area_constraints_eo_takeoff == pytest.approx(1.41, abs=1e-2)
    vt_area_constraints_eo_landing = problem.get_val(
        "data:constraints:vertical_tail:engine_out_landing", units="m**2"
    )
    assert vt_area_constraints_eo_landing == pytest.approx(
        1.41, abs=1e-2
    )  # Should be equal to vtp_area but since
    # the dummy engine is not recognized as an ICE engine the last constraints applies even though it shouldn't


def test_update_ht_area():
    """Tests computation of the horizontal tail area"""

    # Research independent input value in .xml file
    # noinspection PyTypeChecker
    ivc = get_indep_var_comp(
        list_inputs(UpdateHTArea(propulsion_id=ENGINE_WRAPPER)), __file__, XML_FILE
    )

    # Run problem and check obtained value(s) is/(are) correct
    # noinspection PyTypeChecker
    problem = run_system(UpdateHTArea(propulsion_id=ENGINE_WRAPPER), ivc)
    ht_area = problem.get_val("data:geometry:horizontal_tail:area", units="m**2")
    assert ht_area == pytest.approx(3.95, abs=1e-2)


def test_compute_static_margin():
    """Tests computation of static margin"""

    # Research independent input value in .xml file
    ivc = get_indep_var_comp(list_inputs(ComputeStaticMargin()), __file__, XML_FILE)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputeStaticMargin(), ivc)
    stick_fixed_static_margin = problem["data:handling_qualities:stick_fixed_static_margin"]
    assert stick_fixed_static_margin == pytest.approx(0.34, abs=1e-2)
    free_elevator_factor = problem["data:aerodynamics:cruise:neutral_point:free_elevator_factor"]
    assert free_elevator_factor == pytest.approx(0.74, abs=1e-2)
    stick_free_static_margin = problem["data:handling_qualities:stick_free_static_margin"]
    assert stick_free_static_margin == pytest.approx(0.27, abs=1e-2)


def test_compute_to_rotation_limit():
    """Tests computation of static margin"""

    # noinspection PyTypeChecker
    ivc = get_indep_var_comp(
        list_inputs(ComputeTORotationLimitGroup(propulsion_id=ENGINE_WRAPPER)), __file__, XML_FILE
    )

    # Run problem and check obtained value(s) is/(are) correct
    # noinspection PyTypeChecker
    problem = run_system(ComputeTORotationLimitGroup(propulsion_id=ENGINE_WRAPPER), ivc)
    x_cg_rotation_limit = problem["data:handling_qualities:to_rotation_limit:x"]
    assert x_cg_rotation_limit == pytest.approx(1.98, abs=1e-2)
    x_cg_ratio_rotation_limit = problem["data:handling_qualities:to_rotation_limit:MAC_position"]
    assert x_cg_ratio_rotation_limit == pytest.approx(-0.47, abs=1e-2)


def test_compute_balked_landing():
    """Tests computation of static margin"""

    # Research independent input value in .xml file
    # noinspection PyTypeChecker
    ivc = get_indep_var_comp(
        list_inputs(ComputeBalkedLandingLimit(propulsion_id=ENGINE_WRAPPER)), __file__, XML_FILE
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputeBalkedLandingLimit(propulsion_id=ENGINE_WRAPPER), ivc)
    x_cg_balked_landing_limit = problem["data:handling_qualities:balked_landing_limit:x"]
    assert x_cg_balked_landing_limit == pytest.approx(2.67, abs=1e-2)
    x_cg_ratio_balked_landing_limit = problem[
        "data:handling_qualities:balked_landing_limit:MAC_position"
    ]
    assert x_cg_ratio_balked_landing_limit == pytest.approx(0.11, abs=1e-2)


def test_aircraft_modes():
    """
    Testing the module for obtaining the aircraft's modes.
    """

    add_fuselage = False
    use_openvsp = True
    XML_FILE = "cirrus_sr22.xml"

    aircraft_modes(
        add_fuselage,
        use_openvsp,
        XML_FILE
    )


def test_check_modes_complete():
    """
    For testing the complete modes analysis, from calculating the stability derivatives, the aircraft's modes and
    checking them according to hte regulations.
    """

    add_fuselage = False
    use_openvsp = True
    XML_FILE = "cirrus_sr22.xml"

    check_aircraft_modes(
        add_fuselage,
        use_openvsp,
        XML_FILE
    )


def test_short_period_tail_sizing():
    add_fuselage = False
    use_openvsp = True

    short_period_tail_sizing(
        add_fuselage,
        use_openvsp,
        XML_FILE="cirrus_sr22.xml",
        XML_OPTIM_FILE="cirrus_sr22_no_ht_area.xml"
    )


def test_dutch_roll_tail_sizing():
    add_fuselage = False
    use_openvsp = True

    dutch_roll_tail_sizing(
        add_fuselage,
        use_openvsp,
        XML_FILE="cirrus_sr22.xml",
        XML_OPTIM_FILE="cirrus_sr22_dutch_roll.xml"
    )
