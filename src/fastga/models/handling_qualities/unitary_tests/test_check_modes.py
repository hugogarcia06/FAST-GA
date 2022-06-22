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
import openmdao.api as om

import numpy as np

from fastga.models.handling_qualities.check_modes.longitudinal.phugoid_check import CheckPhugoid
from fastga.models.handling_qualities.check_modes.longitudinal.short_period_check import CheckShortPeriod
from tests.testing_utilities import run_system, get_indep_var_comp, list_inputs



def test_check_phugoid():

    ivc = om.IndepVarComp()
    ivc.add_output("data:handling_qualities:longitudinal:modes:phugoid:damping_ratio", val=0.1339)
    ivc.add_output("data:handling_qualities:longitudinal:modes:phugoid:undamped_frequency", val=0.013, units="rad/s")

    # ivc.add_output("data:handling_qualities:longitudinal:modes:phugoid:damping_ratio", val=0.02)
    # ivc.add_output("data:handling_qualities:longitudinal:modes:phugoid:undamped_frequency", val=0.150, units="rad/s")

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(CheckPhugoid(), ivc)


def test_check_short_period():

    ivc = om.IndepVarComp()
    # ivc.add_output("data:handling_qualities:longitudinal:modes:short_period:damping_ratio", val=0.1339)
    # ivc.add_output("data:handling_qualities:longitudinal:modes:short_period:undamped_frequency", val=0.013, units="rad/s")

    ivc.add_output("data:reference_flight_condition:dynamic_pressure", val=92.58, units="slug/(ft*s**2)")
    ivc.add_output("data:handling_qualities:longitudinal:derivatives:CL:alpha", val=5.70, units="rad**-1")
    ivc.add_output("data:geometry:wing:area", val=5500, units="ft**2")
    ivc.add_output("data:reference_flight_condition:weight", val=564032, units="lb")

    ivc.add_output("data:reference_flight_condition:flight_phase_category", val=2.0)
    ivc.add_output("data:handling_qualities:longitudinal:modes:short_period:damping_ratio", val=0.625)
    ivc.add_output("data:handling_qualities:longitudinal:modes:short_period:undamped_frequency", val=0.882, units="rad/s")

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(CheckShortPeriod(), ivc)

def test_figure_digit_short_period_freq():

    n_alpha = 50.0


    z_sp_reqs, wn_sp_reqs = CheckShortPeriod.get_short_period_requirements(2.0, n_alpha)

    z_sp_min_req_1 = z_sp_reqs[0]
    z_sp_max_req_1 = z_sp_reqs[1]
    z_sp_min_req_2 = z_sp_reqs[2]
    z_sp_max_req_2 = z_sp_reqs[3]
    z_sp_min_req_3 = z_sp_reqs[4]

    wn_sp_min_req_1 = wn_sp_reqs[0]
    wn_sp_max_req_1 = wn_sp_reqs[1]
    wn_sp_min_req_2 = wn_sp_reqs[2]
    wn_sp_max_req_2 = wn_sp_reqs[3]
    wn_sp_min_req_3 = wn_sp_reqs[4]

    assert wn_sp_min_req_1 == pytest.approx(0.915, abs=0.05)
    assert wn_sp_max_req_1 == pytest.approx(5.77, abs=0.05)
    assert wn_sp_min_req_2 == pytest.approx(0.6125, abs=0.05)
    assert wn_sp_max_req_2 == pytest.approx(9.8, abs=0.1)

