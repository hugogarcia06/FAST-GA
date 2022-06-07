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

import openmdao.api as om
import numpy as np
import pytest
import math

# Classes to be tested
from fastga.models.handling_qualities.longitudinal_dynamics.components.horizontal_tail.cL_alpha_ht import CLAlphaHT
from fastga.models.handling_qualities.longitudinal_dynamics.components.wing.cL_alpha_wingbody import CLAlphaWingBody
from fastga.models.handling_qualities.lateral_directional_dynamics.components.fuselage.cY_beta_body import CYBetaBody
from fastga.models.handling_qualities.lateral_directional_dynamics.components.fuselage.cn_beta_body import CnBetaBody
from fastga.models.handling_qualities.lateral_directional_dynamics.components.vertical_tail.cY_beta_vt import CYBetaVT
from fastga.models.handling_qualities.lateral_directional_dynamics.components.vertical_tail.cl_beta_vt import ClBetaVT
from fastga.models.handling_qualities.lateral_directional_dynamics.components.vertical_tail.cn_beta_vt import CnBetaVT
from fastga.models.handling_qualities.lateral_directional_dynamics.components.wing.cY_beta_wing import CYBetaWing
from fastga.models.handling_qualities.lateral_directional_dynamics.components.wing.cl_beta_wingbody import ClBetaWingBody
from fastga.models.handling_qualities.lateral_directional_dynamics.components.wing.cn_beta_wing import CnBetaWing
from fastga.models.aerodynamics.components.compute_cnbeta_fuselage import ComputeCnBetaFuselage
from fastga.models.handling_qualities.longitudinal_dynamics.components.horizontal_tail.cL_pitchrate_ht import \
    CLPitchRateHT
from fastga.models.handling_qualities.longitudinal_dynamics.components.horizontal_tail.cm_pitchrate_ht import \
    CmPitchRateHT
from fastga.models.handling_qualities.longitudinal_dynamics.components.wing.cL_pitchrate_wing import CLPitchRateWing
from fastga.models.handling_qualities.longitudinal_dynamics.components.wing.cm_pitchrate_wing import CmPitchRateWing
from fastga.models.handling_qualities.lateral_directional_dynamics.components.vertical_tail.cY_rollrate_vt import CYRollRateVT
from fastga.models.handling_qualities.lateral_directional_dynamics.components.vertical_tail.cl_rollrate_vt import ClRollRateVT
from fastga.models.handling_qualities.lateral_directional_dynamics.components.horizontal_tail.cl_rollrate_ht import ClRollRateHT
from fastga.models.handling_qualities.lateral_directional_dynamics.components.vertical_tail.cn_rollrate_vt import CnRollRateVT
from fastga.models.handling_qualities.lateral_directional_dynamics.components.wing.cY_rollrate_wing import CYRollRateWing
from fastga.models.handling_qualities.lateral_directional_dynamics.components.wing.cl_rollrate_wing import ClRollRateWing
from fastga.models.handling_qualities.lateral_directional_dynamics.components.wing.cn_rollrate_wing import CnRollRateWing
from fastga.models.handling_qualities.lateral_directional_dynamics.components.vertical_tail.cY_yawrate_vt import CYYawRateVT
from fastga.models.handling_qualities.lateral_directional_dynamics.components.vertical_tail.cl_yawrate_vt import ClYawRateVT
from fastga.models.handling_qualities.lateral_directional_dynamics.components.vertical_tail.cn_yawrate_vt import CnYawRateVT
from fastga.models.handling_qualities.lateral_directional_dynamics.components.wing.cl_yawrate_wing import ClYawRateWing
from fastga.models.handling_qualities.lateral_directional_dynamics.components.wing.cn_yawrate_wing import CnYawRateWing

from tests.testing_utilities import run_system, get_indep_var_comp, list_inputs


def test_CL_alpha_wing():
    """
    Test data from USAF DATCOM page 1228/3070.
    """
    A_W = 8.0
    A_W_e = 7.61
    taper_ratio_W = 0.45
    taper_ratio_W_e = 0.47
    sweep_0_W = 46.3
    # b/d = 10.0
    d = 0.5
    b_W = 5.0

    cl_alpha_w = 6.30
    ivc = om.IndepVarComp()
    # Reference Flight Condition
    ivc.add_output("data:reference_flight_condition:mach", val=0.19)

    # Wing Geometry
    ivc.add_output("data:geometry:wing:aspect_ratio", val=A_W_e)
    ivc.add_output("data:geometry:wing:taper_ratio", val=taper_ratio_W)
    ivc.add_output("data:geometry:wing:span", val=b_W, units="ft")
    ivc.add_output("data:geometry:wing:sweep_0", val=sweep_0_W, units="deg")

    # Fuselage Geometry
    ivc.add_output("data:geometry:fuselage:maximum_height", val=d, units="ft")
    ivc.add_output("data:geometry:fuselage:maximum_width", val=d, units="ft")

    # Wing Aerodynamics
    ivc.add_output("data:aerodynamics:wing:airfoil:CL_alpha", val=cl_alpha_w, units="rad**-1")

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(CLAlphaWingBody(), ivc)
    CL_alpha_WB = problem.get_val("data:handling_qualities:longitudinal:derivatives:wing:CL:alpha", units="rad**-1")
    assert CL_alpha_WB == pytest.approx(4.286, abs=0.35)


def test_CL_alpha_ht():
    """
    Test data from USAF DATCOM page 1228/3070.
    """
    A_W = 8.0
    A_W_e = 7.61
    S_W = 100.0
    S_W_e = 89.6
    taper_ratio_W = 0.45
    taper_ratio_W_e = 0.47
    sweep_0_W = 46.3
    sweep_0_W = sweep_0_W * math.pi / 180.0
    sweep_25_W = math.atan(math.tan(sweep_0_W) - 4 / A_W * (0.25 * ((1 - taper_ratio_W) / (1 + taper_ratio_W))))
    # b/d = 10.0
    d = 0.5
    b_W = 5.0

    S_H = 16.0
    S_H_e = 11.76
    b_H = 1.414
    A_H = 4.0
    A_H_e = 3.45
    sweep_0_H = 47.6
    taper_ratio_H = 0.45
    taper_ratio_H_e = 0.507
    l_H = 0.72
    h_H = -0.015

    cl_alpha_w = 6.30
    ivc = om.IndepVarComp()
    # Reference Flight Condition
    ivc.add_output("data:reference_flight_condition:mach", val=0.19)

    # Wing Geometry
    ivc.add_output("data:geometry:wing:area", val=S_W, units="ft**2")
    ivc.add_output("data:geometry:wing:aspect_ratio", val=A_W)
    ivc.add_output("data:geometry:wing:taper_ratio", val=taper_ratio_W)
    ivc.add_output("data:geometry:wing:span", val=b_W, units="ft")
    ivc.add_output("data:geometry:wing:sweep_25", val=sweep_25_W, units="rad")
    ivc.add_output("data:geometry:wing:sweep_0", val=sweep_0_W, units="rad")

    # Wing Aerodynamics
    ivc.add_output("data:aerodynamics:wing:airfoil:CL_alpha", val=cl_alpha_w, units="rad**-1")

    # Horizontal Tail Geometry
    ivc.add_output("data:geometry:horizontal_tail:area", val=S_H_e, units="ft**2")
    ivc.add_output("data:geometry:horizontal_tail:span", val=b_H, units="ft")
    ivc.add_output("data:geometry:has_T_tail", val=np.nan)
    ivc.add_output("data:geometry:horizontal_tail:aspect_ratio", val=A_H_e)
    ivc.add_output("data:geometry:horizontal_tail:taper_ratio", val=taper_ratio_H_e)
    ivc.add_output("data:geometry:horizontal_tail:MAC:at25percent:x:from_wingMAC25", val=l_H, units="ft")
    ivc.add_output("data:geometry:horizontal_tail:sweep_0", val=sweep_0_H, units="deg")
    ivc.add_output("data:geometry:horizontal_tail:z:from_wingMAC25", val=h_H, units="ft")

    # Horizontal Tail Aerodynamics
    ivc.add_output("data:aerodynamics:horizontal_tail:airfoil:CL_alpha", val=cl_alpha_w, units="rad**-1")
    ivc.add_output("data:aerodynamics:horizontal_tail:efficiency", val=0.9)

    # Fuselage Geometry
    ivc.add_output("data:geometry:fuselage:maximum_height", val=d, units="ft")
    ivc.add_output("data:geometry:fuselage:maximum_width", val=d, units="ft")

    # Vertical tail Geometry
    ivc.add_output("data:geometry:vertical_tail:span", val=np.nan, units="ft")

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(CLAlphaHT(), ivc)
    CL_alpha_H = problem.get_val("data:handling_qualities:longitudinal:derivatives:wing:CL:alpha", units="rad**-1")
    assert CL_alpha_H == pytest.approx(4.286, abs=0.35)


def test_CY_beta_vt():
    """
    Test computation of the side force coefficient due to sideslip.
    """

    # Define input values calculated from other modules. These values correspond to the value of the Sample Problem in
    # USAF DATCOM in page 1751/3070 (pdf)
    ivc = om.IndepVarComp()
    ivc.add_output("data:reference_flight_condition:mach", 0.6)

    ivc.add_output("data:geometry:wing:area", 36.0, units="inch**2")
    ivc.add_output("data:geometry:wing:aspect_ratio", 4.0)
    ivc.add_output("data:geometry:wing:sweep_25", 45.0, units="deg")
    ivc.add_output("data:geometry:wing:vertical_position", 0.0, units="m")

    ivc.add_output("data:geometry:vertical_tail:area", 5.94, units="inch**2")
    ivc.add_output("data:geometry:vertical_tail:span", 3.30, units="inch")
    ivc.add_output("data:geometry:vertical_tail:aspect_ratio", 1.835)
    ivc.add_output("data:geometry:vertical_tail:taper_ratio", 1.0)
    ivc.add_output("data:geometry:vertical_tail:sweep_50", 0.0, units="deg")
    ivc.add_output("data:geometry:vertical_tail:sweep_0", val=np.nan, units="deg")
    ivc.add_output("data:geometry:vertical_tail:root:chord", val=np.nan, units="m")

    ivc.add_output("data:aerodynamics:vertical_tail:k_ar_effective", 2.61)
    ivc.add_output("data:aerodynamics:vertical_tail:airfoil:CL_alpha", 6.57, units="rad**-1")

    ivc.add_output("data:geometry:fuselage:length", 18.25, units="inch")
    ivc.add_output("data:geometry:fuselage:rear_length", 18.25, units="inch")
    ivc.add_output("data:geometry:fuselage:maximum_width", 1.667, units="inch")
    ivc.add_output("data:geometry:fuselage:maximum_height", 1.667, units="inch")
    ivc.add_output("data:geometry:fuselage:depth_quarter_vt", 1.667, units="inch")

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(CYBetaVT(), ivc)
    CY_beta_V = problem.get_val("data:handling_qualities:lateral:derivatives:vertical_tail:CY:beta", units="rad**-1")
    assert CY_beta_V == pytest.approx(-0.645, abs=0.1)


def test_CY_beta_wing():
    """
    Test computation of the side force coefficient due to sideslip.
    """

    # Define input values calculated from other modules. These values correspond to the value of the Sample Problem in
    # USAF DATCOM in page 1751/3070 (pdf)
    ivc = om.IndepVarComp()

    ivc.add_output("data:geometry:wing:dihedral", 0.0, units="inch**2")

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(CYBetaWing(), ivc)
    CY_beta_W = problem.get_val("data:handling_qualities:lateral:derivatives:wing:CY:beta", units="rad**-1")
    assert CY_beta_W == pytest.approx(0.0, abs=0.001)


def test_CY_beta_body():
    """
    Test computation of the side force coefficient due to sideslip. Data from USAF DATCOM (pdf) page 1751/3070.
    """

    # Define input values calculated from other modules. These values correspond to the value of the Sample Problem in
    # USAF DATCOM in page 1751/3070 (pdf)
    ivc = om.IndepVarComp()
    ivc.add_output("data:geometry:wing:area", 36.0, units="inch**2")
    ivc.add_output("data:geometry:wing:vertical_position", 0.0, units="m")

    ivc.add_output("data:geometry:fuselage:maximum_width", 1.667, units="inch")
    ivc.add_output("data:geometry:fuselage:maximum_height", 1.667, units="inch")
    ivc.add_output("data:geometry:fuselage:length", 18.25, units="inch")
    ivc.add_output("data:geometry:fuselage:volume", 33.54, units="inch**3")

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(CYBetaBody(), ivc)
    CY_beta_B = problem.get_val("data:handling_qualities:lateral:derivatives:fuselage:CY:beta", units="rad**-1")
    assert CY_beta_B == pytest.approx(-0.115, abs=0.01)


def test_Cl_beta_vt():
    """

    """
    ivc = om.IndepVarComp()
    ivc.add_output("data:reference_flight_condition:alpha", 3.20, units="deg")

    ivc.add_output("data:geometry:wing:span", 12.0, units="inch")

    ivc.add_output("data:geometry:vertical_tail:MAC:z", 1.65, units="inch")
    ivc.add_output("data:geometry:vertical_tail:MAC:at25percent:x:from_wingMAC25", 6.035, units="inch")
    ivc.add_output("data:handling_qualities:lateral:derivatives:vertical_tail:CY:beta", -0.01125, units="deg**-1")

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ClBetaVT(), ivc)
    Cl_beta_V = problem.get_val("data:handling_qualities:lateral:derivatives:vertical_tail:Cl:beta", units="rad**-1")
    assert Cl_beta_V == pytest.approx(-0.0704, abs=0.005)


def test_Cl_beta_wingbody_one():
    """
    Data from USAF DATCOM (pdf) in page 1784/3070.
    """
    ivc = om.IndepVarComp()
    ivc.add_output("data:reference_flight_condition:CL", 0.2)
    ivc.add_output("data:reference_flight_condition:mach", 0.6)

    ivc.add_output("data:geometry:wing:span", 12.0, units="inch")
    ivc.add_output("data:geometry:wing:aspect_ratio", 4.0)
    ivc.add_output("data:geometry:wing:taper_ratio", 0.3)
    ivc.add_output("data:geometry:wing:sweep_50", 40.9, units="deg")
    ivc.add_output("data:geometry:wing:sweep_25", 0.0, units="deg")
    ivc.add_output("data:geometry:wing:dihedral", 0.0, units="deg")
    ivc.add_output("data:geometry:wing:twist", 0.0, units="deg")
    ivc.add_output("data:geometry:wing:vertical_position", 0.0, units="inch")
    ivc.add_output(
        "data:geometry:wing:tip:half_chord_point:x:absolut""data:geometry:wing:vertical_position", 14.3, units="inch"
    )
    ivc.add_output("data:geometry:fuselage:maximum_width", 6.70, units="inch")
    ivc.add_output("data:geometry:fuselage:maximum_height", 6.70, units="inch")

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ClBetaWingBody(), ivc)
    Cl_beta_WB = problem.get_val("data:handling_qualities:lateral:derivatives:wing-body:Cl:beta", units="rad**-1")
    assert Cl_beta_WB == pytest.approx(-0.00123, abs=0.001)


def test_Cl_beta_wingbody_two():
    """
    Data from USAF DATCOM (pdf) in page 1546/3070.
    """
    ivc = om.IndepVarComp()
    ivc.add_output("data:reference_flight_condition:CL", 0.2)
    ivc.add_output("data:reference_flight_condition:mach", 0.105)

    ivc.add_output("data:geometry:wing:span", 60.0, units="inch")
    ivc.add_output("data:geometry:wing:aspect_ratio", 6.383)
    ivc.add_output("data:geometry:wing:taper_ratio", 1.0)
    ivc.add_output("data:geometry:wing:sweep_50", 0.0, units="deg")
    ivc.add_output("data:geometry:wing:sweep_25", 0.0, units="deg")
    ivc.add_output("data:geometry:wing:dihedral", 5.0, units="deg")
    ivc.add_output("data:geometry:wing:twist", 0.0, units="deg")
    ivc.add_output("data:geometry:wing:vertical_position", -2.66, units="inch")
    ivc.add_output(
        "data:geometry:wing:tip:half_chord_point:x:absolut", 15.7, units="inch"
    )
    ivc.add_output("data:geometry:wing:tip:leading_edge:x:local", val=np.nan, units="inch")
    ivc.add_output("data:geometry:wing:root:leading_edge:x:absolut", val=np.nan, units="inch")
    ivc.add_output("data:geometry:wing:tip:chord", val=np.nan, units="inch")

    ivc.add_output("data:geometry:fuselage:maximum_width", 6.70, units="inch")
    ivc.add_output("data:geometry:fuselage:maximum_height", 6.70, units="inch")

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ClBetaWingBody(), ivc)
    Cl_beta_WB = problem.get_val("data:handling_qualities:lateral:derivatives:wing-body:Cl:beta", units="deg**-1")
    assert Cl_beta_WB == pytest.approx(-0.00207, abs=0.0005)


def test_Cl_beta_ht():
    """
    There is no sample problem for the horizontal tail in the DATCOM since it is neglected. However, the horizontal tail
    can be treated as a smaller wing, thus, its coefficient is calculated in the exact same way.
    """


def test_Cn_beta_vt():
    """
    Data from USAF DATCOM (pdf) in page 1797/3070.
    """
    ivc = om.IndepVarComp()
    ivc.add_output("data:reference_flight_condition:alpha", 0.0, units="rad")

    ivc.add_output("data:geometry:wing:span", 12.0, units="inch")

    ivc.add_output("data:geometry:vertical_tail:MAC:z", 1.65, units="inch")
    ivc.add_output("data:geometry:vertical_tail:MAC:at25percent:x:from_wingMAC25", 6.035, units="inch")
    ivc.add_output("data:handling_qualities:lateral:derivatives:vertical_tail:CY:beta", -0.645, units="rad**-1")

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(CnBetaVT(), ivc)
    Cn_beta_V = problem.get_val("data:handling_qualities:lateral:derivatives:vertical_tail:Cn:beta", units="rad**-1")
    assert Cn_beta_V == pytest.approx(0.324, abs=0.05)


def test_Cn_beta_body():
    """
    In the case of the fuselage contribution to Cn_beta, the value used is already computed by previous FAST-OAD
    modules.
    Data test from USAF DATCOM page 1797/3070
    """
    ivc = om.IndepVarComp()
    d = 0.416
    ivc.add_output("data:geometry:fuselage:maximum_width", val=d, units="ft")
    ivc.add_output("data:geometry:fuselage:maximum_height", val=d, units="ft")
    ivc.add_output("data:geometry:fuselage:length", val=4.1, units="ft")
    lav = 1.6 * d
    lar = 3.15 * d
    ivc.add_output("data:geometry:fuselage:front_length", val=lav, units="ft")
    ivc.add_output("data:geometry:fuselage:rear_length", val=lar, units="ft")
    ivc.add_output("data:geometry:wing:area", val=2.25, units="ft**2")
    ivc.add_output("data:geometry:wing:span", val=3.0, units="ft")


    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputeCnBetaFuselage(), ivc)
    Cn_beta_B = problem.get_val("data:aerodynamics:fuselage:cruise:CnBeta", units="deg**-1")
    assert Cn_beta_B == pytest.approx(-0.00135, abs=0.0005)


def test_CL_pitchrate_wing():
    """
    Test data from USAF DATCOM page 2407/3070.
    """
    ivc = om.IndepVarComp()
    # Reference Flight Condition
    ivc.add_output("data:reference_flight_condition:mach", val=0.2)

    # Wing Geometry
    ar = 4.0
    tr = 0.68
    sweep_0 = 46.3
    sweep_0 = 46.3 * math.pi / 180.0
    sweep_25 = math.atan(math.tan(sweep_0) - 4 / ar * (0.25 * ((1 - tr)/(1 + tr))))
    sweep_50 = math.atan(math.tan(sweep_0) - 4 / ar * (0.5 * ((1 - tr)/(1 + tr))))
    ivc.add_output("data:geometry:wing:aspect_ratio", val=ar)
    ivc.add_output("data:geometry:wing:taper_ratio", val=tr)
    ivc.add_output("data:geometry:wing:sweep_0", val=sweep_0, units="rad")
    ivc.add_output("data:geometry:wing:sweep_25", val=sweep_25, units="rad")
    ivc.add_output("data:geometry:wing:sweep_50", val=sweep_50, units="rad")
    ivc.add_output("data:geometry:wing:MAC:length", val=1.3, units="ft")

    # Wing Aerodynamics
    ivc.add_output("data:aerodynamics:wing:airfoil:CL_alpha", val=6.19, units="rad**-1")

    ivc.add_output("data:handling_qualities:stick_fixed_static_margin", val=0.0118)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(CLPitchRateWing(), ivc)
    CL_q_W = problem.get_val("data:handling_qualities:longitudinal:derivatives:wing:CL:pitchrate", units="rad**-1")
    assert CL_q_W == pytest.approx(0.129, abs=0.01)


def test_CL_pitchrate_ht():
    """
    Test data from USAF DATCOM page 2669/3070.
    """
    ivc = om.IndepVarComp()
    # Reference Flight Condition
    ivc.add_output("data:reference_flight_condition:mach", val=0.6)

    # Wing Geometry
    ivc.add_output("data:geometry:wing:area", val=414.0, units="ft**2")
    ivc.add_output("data:geometry:wing:MAC:length", val=12.133, units="ft")

    # Horizontal Tail Geometry
    ivc.add_output("data:geometry:horizontal_tail:area", val=99.16, units="ft**2")
    ivc.add_output("data:geometry:horizontal_tail:span", val=19.915, units="ft")
    ivc.add_output("data:geometry:horizontal_tail:aspect_ratio", val=4.0)
    ivc.add_output("data:geometry:horizontal_tail:taper_ratio", val=0.0)
    ivc.add_output("data:geometry:horizontal_tail:MAC:at25percent:x:from_wingMAC25", val=27.42, units="ft")
    ivc.add_output("data:geometry:horizontal_tail:sweep_0", val=np.nan, units="deg")
    ivc.add_output("data:geometry:horizontal_tail:sweep_50", val=26.56, units="deg")

    # Fuselage Geometry
    ivc.add_output("data:geometry:fuselage:maximum_height", val=7.127, units="ft")
    ivc.add_output("data:geometry:fuselage:maximum_width", val=7.127, units="ft")

    # Horizontal Tail Aerodynamics
    ivc.add_output("data:aerodynamics:horizontal_tail:airfoil:CL_alpha", val=6.19, units="rad**-1")
    ivc.add_output("data:aerodynamics:horizontal_tail:efficiency", val=0.9)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(CLPitchRateHT(), ivc)
    CL_q_H = problem.get_val("data:handling_qualities:longitudinal:derivatives:horizontal_tail:CL:pitchrate", units="rad**-1")
    assert CL_q_H == pytest.approx(0.129, abs=0.01)


def test_Cm_pitchrate_wing():
    """
    Test data from USAF DATCOM page 2425/3070.
    """
    ivc = om.IndepVarComp()
    # Reference Flight Condition
    ivc.add_output("data:reference_flight_condition:mach", val=0.6)

    # Wing Geometry
    ivc.add_output("data:geometry:wing:aspect_ratio", val=4.0)
    ivc.add_output("data:geometry:wing:sweep_25", val=45.0, units="deg")
    ivc.add_output("data:geometry:wing:MAC:length", val=1.5, units="ft")

    # Wing Aerodynamics
    ivc.add_output("data:aerodynamics:wing:airfoil:CL_alpha", val=0.109, units="deg**-1")

    ivc.add_output("data:handling_qualities:stick_fixed_static_margin", val=0.0118)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(CmPitchRateWing(), ivc)
    Cm_q_W = problem.get_val("data:handling_qualities:longitudinal:derivatives:wing:Cm:pitchrate", units="rad**-1")
    assert Cm_q_W == pytest.approx(0.129, abs=0.01)


def test_Cm_pitchrate_ht():
    """
    Test data from USAF DATCOM page 2680/3070.
    """
    S_W = 414.0
    mac_W = 12.133
    S_H = 99.16
    A_H = 4.0
    taper_ratio_H = 0.0
    l_H = 27.42 # l_H/mac_W = 2.26
    sweep_0_H = 45.0
    sweep_50_H =  26.56
    cl_alpha_h = 6.19

    ivc = om.IndepVarComp()
    # Reference Flight Condition
    ivc.add_output("data:reference_flight_condition:mach", val=0.6)

    # Wing Geometry
    ivc.add_output("data:geometry:wing:area", val=S_W, units="ft**2")
    ivc.add_output("data:geometry:wing:MAC:length", val=mac_W, units="ft")

    # Horizontal Tail Geometry
    ivc.add_output("data:geometry:horizontal_tail:area", val=S_H, units="ft**2")
    ivc.add_output("data:geometry:horizontal_tail:aspect_ratio", val=A_H)
    ivc.add_output("data:geometry:horizontal_tail:taper_ratio", val=taper_ratio_H)
    ivc.add_output("data:geometry:horizontal_tail:MAC:at25percent:x:from_wingMAC25", val=l_H, units="ft")
    ivc.add_output("data:geometry:horizontal_tail:sweep_0", val=sweep_0_H, units="deg")
    ivc.add_output("data:geometry:horizontal_tail:sweep_50", val=sweep_50_H, units="deg")

    # Horizontal Tail Aerodynamics
    ivc.add_output("data:aerodynamics:horizontal_tail:airfoil:CL_alpha", val=cl_alpha_h, units="rad**-1")
    ivc.add_output("data:aerodynamics:horizontal_tail:efficiency", val=0.9)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(CmPitchRateHT(), ivc)
    Cm_q_H = problem.get_val("data:handling_qualities:longitudinal:derivatives:horizontal_tail:Cm:pitchrate",
                             units="rad**-1")
    assert Cm_q_H == pytest.approx(0.129, abs=0.01)


def test_CY_rollrate_vt():
    """
    Test data from USAF DATCOM page 2714/3070.
    """
    ivc = om.IndepVarComp()
    ivc.add_output("data:reference_flight_condition:mach", 0.17)
    ivc.add_output("data:reference_flight_condition:alpha", 8.0, units="deg")

    ivc.add_output("data:geometry:wing:span", 38.84, units="inch")
    ivc.add_output("data:geometry:wing:sweep_25", 36.2, units="deg")

    ivc.add_output("data:geometry:vertical_tail:MAC:z", 5.0, units="inch")
    ivc.add_output("data:geometry:vertical_tail:MAC:at25percent:x:from_wingMAC25", 24.3, units="inch")
    ivc.add_output("data:handling_qualities:lateral:derivatives:vertical_tail:CY:beta", -0.729, units="rad**-1")

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(CYRollRateVT(), ivc)
    CY_p_V = problem.get_val("data:handling_qualities:lateral:derivatives:vertical_tail:CY:rollrate",
                             units="rad**-1")
    assert CY_p_V == pytest.approx(0.129, abs=0.01)


def test_CY_rollrate_wing():
    """
    Test data from USAF DATCOM page /3070.
    """
    ivc = om.IndepVarComp()
    # Flight reference condition
    ivc.add_output("data:reference_flight_condition:mach", val=0.7)
    ivc.add_output("data:reference_flight_condition:alpha", val=np.nan, units="rad")

    # Wing Geometry
    ivc.add_output("data:geometry:wing:span", val=np.nan, units="m")
    ivc.add_output("data:geometry:wing:aspect_ratio", val=np.nan)
    ivc.add_output("data:geometry:wing:taper_ratio", val=np.nan)
    ivc.add_output("data:geometry:wing:dihedral", val=np.nan, units="rad")
    ivc.add_output("data:geometry:wing:sweep_25", val=np.nan, units="rad")

    # Wing Aerodynamics
    ivc.add_output("data:aerodynamics:wing:cruise:induced_drag_coefficient", val=np.nan)

    # Vertical tail Geometry
    ivc.add_output("data:geometry:vertical_tail:MAC:z", val=np.nan, units="m")
    ivc.add_output("data:geometry:vertical_tail:MAC:at25percent:x:from_wingMAC25", val=np.nan, units="m")

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(CYRollRateWing(), ivc)
    CY_p_W = problem.get_val("data:handling_qualities:lateral:derivatives:wing:CY:rollrate",
                             units="rad**-1")
    assert CY_p_W == pytest.approx(0.129, abs=0.01)


def test_Cl_rollrate_vt():
    """
    Test data from USAF DATCOM page 2721/3070.
    """

    ivc = om.IndepVarComp()
    ivc.add_output("data:reference_flight_condition:alpha", 4.0, units="deg")

    ivc.add_output("data:geometry:wing:span", 2.16, units="ft")

    ivc.add_output("data:geometry:vertical_tail:MAC:z", 0.308, units="ft")
    ivc.add_output("data:geometry:vertical_tail:MAC:at25percent:x:from_wingMAC25", 1.25, units="ft")
    ivc.add_output("data:handling_qualities:lateral:derivatives:vertical_tail:CY:beta", -0.646, units="rad**-1")

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ClRollRateVT(), ivc)
    Cl_p_V = problem.get_val("data:handling_qualities:lateral:derivatives:vertical_tail:Cl:rollrate",
                             units="rad**-1")
    assert Cl_p_V == pytest.approx(-0.00536, abs=0.0005)


def test_Cl_rollrate_ht():
    """
    Test data from USAF DATCOM page 2721/3070.
    """

    ivc = om.IndepVarComp()
    ivc.add_output("data:reference_flight_condition:mach", 0.25)

    ivc.add_output("data:geometry:wing:span", 2.16, units="ft")
    ivc.add_output("data:geometry:wing:taper_ratio", 0.38)
    ivc.add_output("data:geometry:wing:area", 1.90, units="ft**2")

    ivc.add_output("data:geometry:horizontal_tail:aspect_ratio", val=2.97)
    ivc.add_output("data:geometry:horizontal_tail:area", val=0.48, units="ft**2")
    ivc.add_output("data:geometry:horizontal_tail:span", val=1.20, units="ft")
    ivc.add_output("data:geometry:horizontal_tail:sweep_25", val=10.5, units="deg")

    ivc.add_output("data:aerodynamics:horizontal_tail:cruise:induced_drag_coefficient", val=1.0, units="rad**-1")

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ClRollRateHT(), ivc)
    Cl_p_H = problem.get_val("data:handling_qualities:lateral:derivatives:horizontal_tail:Cl:rollrate",
                             units="rad**-1")
    assert Cl_p_H == pytest.approx(-0.00986, abs=0.0005)


def test_Cl_rollrate_wing_one():
    """
    Test data from USAF DATCOM page 2721/3070. Using the data from the horizontal tail (same process).
    """

    ivc = om.IndepVarComp()
    ivc.add_output("data:reference_flight_condition:mach", 0.25)
    ivc.add_output("data:reference_flight_condition:CL", 0.3)
    ivc.add_output("data:reference_flight_condition:CD0", 0.00927)

    # Wing Geometry
    ivc.add_output("data:geometry:wing:span", val=1.20, units="ft")
    ivc.add_output("data:geometry:wing:aspect_ratio", val=2.97)
    ivc.add_output("data:geometry:wing:taper_ratio", val=0.31)
    ivc.add_output("data:geometry:wing:dihedral", val=0.0, units="deg")
    ivc.add_output("data:geometry:wing:sweep_25", val=10.5, units="deg")
    ivc.add_output("data:geometry:wing:sweep_0", val=np.nan, units="deg")
    ivc.add_output("data:geometry:wing:vertical_position", val=-8.30, units="inch")

    # Wing Aerodynamics
    ivc.add_output("data:aerodynamics:wing:cruise:induced_drag_coefficient", val=1.0)
    ivc.add_output("data:aerodynamics:wing:airfoil:CL_alpha", val=np.nan, units="rad**-1")

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ClRollRateWing(), ivc)
    Cl_p_W = problem.get_val("data:handling_qualities:lateral:derivatives:wing:Cl:rollrate", units="rad**-1")
    assert Cl_p_W == pytest.approx(-0.30, abs=0.05)


def test_Cl_rollrate_wing_two():
    """
    Test data from USAF DATCOM page 2470/3070.
    """

    ivc = om.IndepVarComp()
    ivc.add_output("data:reference_flight_condition:mach", 0.13)
    ivc.add_output("data:reference_flight_condition:CL", 0.3)
    ivc.add_output("data:reference_flight_condition:CD0", 0.036)

    # Wing Geometry
    ivc.add_output("data:geometry:wing:span", val=2.16, units="ft")
    ivc.add_output("data:geometry:wing:aspect_ratio", val=3.0)
    ivc.add_output("data:geometry:wing:taper_ratio", val=0.15)
    ivc.add_output("data:geometry:wing:dihedral", val=0.0, units="deg")
    ivc.add_output("data:geometry:wing:sweep_25", val=36.9, units="deg")
    ivc.add_output("data:geometry:wing:sweep_0", val=np.nan, units="deg")
    ivc.add_output("data:geometry:wing:vertical_position", val=0.0, units="ft")

    # Wing Aerodynamics
    ivc.add_output("data:aerodynamics:wing:cruise:induced_drag_coefficient", val=0.883)
    ivc.add_output("data:aerodynamics:wing:airfoil:CL_alpha", val=np.nan, units="rad**-1")

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ClRollRateWing(), ivc)
    Cl_p_W = problem.get_val("data:handling_qualities:lateral:derivatives:wing:Cl:rollrate", units="rad**-1")
    assert Cl_p_W == pytest.approx(-0.2328, abs=0.03)


def test_Cn_rollrate_vt():
    """
    Test data from USAF DATCOM page 2731/3070.
    """

    ivc = om.IndepVarComp()
    ivc.add_output("data:reference_flight_condition:alpha", 6.0, units="deg")

    ivc.add_output("data:geometry:wing:span", 2.16, units="ft")

    ivc.add_output("data:geometry:vertical_tail:MAC:z", 0.308, units="ft")
    ivc.add_output("data:geometry:vertical_tail:MAC:at25percent:x:from_wingMAC25", 1.25, units="ft")
    ivc.add_output("data:handling_qualities:lateral:derivatives:vertical_tail:CY:beta", -0.646, units="rad**-1")

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(CnRollRateVT(), ivc)
    Cn_p_V = problem.get_val("data:handling_qualities:lateral:derivatives:vertical_tail:Cn:rollrate",
                             units="rad**-1")
    assert Cn_p_V == pytest.approx(-0.0466, abs=0.001)


def test_Cn_rollrate_wing():
    """
    Test data from USAF DATCOM page 2497/3070.
    """

    ivc = om.IndepVarComp()
    ivc.add_output("data:reference_flight_condition:mach", val=0.70)
    ivc.add_output("data:reference_flight_condition:CL", val=0.1)
    ivc.add_output("data:reference_flight_condition:flaps_deflection", val=np.nan)

    # Wing Geometry
    ivc.add_output("data:geometry:wing:span", val=3.0, units="ft")
    ivc.add_output("data:geometry:wing:MAC:length", val=5.0, units="m")
    ivc.add_output("data:geometry:wing:aspect_ratio", val=4.0)
    ivc.add_output("data:geometry:wing:taper_ratio", val=0.6)
    ivc.add_output("data:geometry:wing:twist", val=0.0, units="deg")
    ivc.add_output("data:geometry:wing:sweep_25", val=45.0, units="deg")
    # flaps
    ivc.add_output("data:geometry:flap:span_ratio", val=np.nan)

    ivc.add_output("data:handling_qualities:stick_fixed_static_margin", val=0.0)

    # Wing Aerodynamics
    ivc.add_output("data:aerodynamics:wing:airfoil:CL_alpha", val=np.nan, units="rad**-1")

    ivc.add_output("data:aerodynamics:flaps:landing:CL_2D", val=np.nan)
    ivc.add_output("data:aerodynamics:flaps:takeoff:CL_2D", val=np.nan)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(CnRollRateWing(), ivc)
    Cn_p_W = problem.get_val("data:handling_qualities:lateral:derivatives:wing:Cn:beta", units="rad**-1")
    assert Cn_p_W == pytest.approx(-0.0152, abs=0.001)


def test_CY_yawrate_vt():
    """

    """

    ivc = om.IndepVarComp()
    ivc.add_output("data:reference_flight_condition:alpha", 6.0, units="deg")

    ivc.add_output("data:geometry:wing:span", 38.84, units="inch")

    ivc.add_output("data:geometry:vertical_tail:MAC:z", 5.0, units="inch")
    ivc.add_output("data:geometry:vertical_tail:MAC:at25percent:x:from_wingMAC25", 24.3, units="inch")
    ivc.add_output("data:handling_qualities:lateral:derivatives:vertical_tail:CY:beta", -0.64, units="rad**-1")

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(CYYawRateVT(), ivc)
    CY_r_V = problem.get_val("data:handling_qualities:lateral:derivatives:vertical_tail:CY:yawrate",
                             units="rad**-1")
    assert CY_r_V == pytest.approx(0.84, abs=0.05)


def test_Cl_yawrate_vt():
    """

    """

    ivc = om.IndepVarComp()
    ivc.add_output("data:reference_flight_condition:alpha", 6.0, units="deg")

    ivc.add_output("data:geometry:wing:span", 38.84, units="inch")

    ivc.add_output("data:geometry:vertical_tail:MAC:z", 5.0, units="inch")
    ivc.add_output("data:geometry:vertical_tail:MAC:at25percent:x:from_wingMAC25", 24.3, units="inch")
    ivc.add_output("data:handling_qualities:lateral:derivatives:vertical_tail:CY:beta", -0.64, units="rad**-1")

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ClYawRateVT(), ivc)
    Cl_r_V = problem.get_val("data:handling_qualities:lateral:derivatives:vertical_tail:Cl:yawrate",
                             units="rad**-1")
    assert Cl_r_V == pytest.approx(0.051, abs=0.005)


def test_Cl_yawrate_wing():
    """
    Data test from USAF DATCOM (pdf) page 2519/3070
    """

    ivc = om.IndepVarComp()
    # Flight reference condition
    ivc.add_output("data:reference_flight_condition:mach", val=0.0)
    ivc.add_output("data:reference_flight_condition:CL", val=0.2)
    ivc.add_output("data:reference_flight_condition:flaps_deflection", val=np.nan)

    # Wing Geometry
    ivc.add_output("data:geometry:wing:aspect_ratio", val=2.61)
    ivc.add_output("data:geometry:wing:taper_ratio", val=1.0)
    ivc.add_output("data:geometry:wing:dihedral", val=10, units="deg")
    ivc.add_output("data:geometry:wing:twist", val=0, units="deg")
    ivc.add_output("data:geometry:wing:sweep_25", val=45, units="deg")
    ivc.add_output("data:geometry:flap:flap_location", val=np.nan)

    ivc.add_output("data:aerodynamics:flaps:landing:CL_2D", val=np.nan)
    ivc.add_output("data:aerodynamics:flaps:takeoff:CL_2D", val=np.nan)

    # Wing Aerodynamics
    ivc.add_output("data:aerodynamics:wing:airfoil:CL_alpha", val=np.nan, units="rad**-1")

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ClYawRateWing(), ivc)
    Cl_r_W = problem.get_val("data:handling_qualities:lateral:derivatives:wing:Cl:yawrate", units="rad**-1")
    assert Cl_r_W == pytest.approx(0.1452, abs=0.01)

def test_Cn_yawrate_vt():
    """
    Test data from USAF DATCOM (pdf) page 2742/3070.
    """

    ivc = om.IndepVarComp()
    ivc.add_output("data:reference_flight_condition:alpha", 6.0, units="deg")

    ivc.add_output("data:geometry:wing:span", 38.84, units="inch")

    ivc.add_output("data:geometry:vertical_tail:MAC:z", 5.0, units="inch")
    ivc.add_output("data:geometry:vertical_tail:MAC:at25percent:x:from_wingMAC25", 24.3, units="inch")
    ivc.add_output("data:handling_qualities:lateral:derivatives:vertical_tail:CY:beta", -0.64, units="rad**-1")

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(CnYawRateVT(), ivc)
    Cn_r_V = problem.get_val("data:handling_qualities:lateral:derivatives:vertical_tail:Cn:yawrate",
                             units="rad**-1")
    assert Cn_r_V == pytest.approx(-0.578, abs=0.07)

def test_Cn_yawrate_wing_one():
    """
    Data test from USAF DATCOM (pdf) page 2530/3070
    """

    ivc = om.IndepVarComp()
    # Flight reference condition
    ivc.add_output("data:reference_flight_condition:CL", val=0.1)
    ivc.add_output("data:reference_flight_condition:CD0", val=0.0186)

    # Wing Geometry
    ivc.add_output("data:geometry:wing:aspect_ratio", val=2.31)
    ivc.add_output("data:geometry:wing:taper_ratio", val=0.0)
    ivc.add_output("data:geometry:wing:MAC:length", val=0.5, units="ft")
    ivc.add_output("data:geometry:wing:sweep_25", val=52.4, units="deg")

    ivc.add_output("data:handling_qualities:stick_fixed_static_margin", val=0.0)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(CnYawRateWing(), ivc)
    Cn_r_W = problem.get_val("data:handling_qualities:lateral:derivatives:wing:Cn:yawrate", units="rad**-1")
    assert Cn_r_W == pytest.approx(-0.0126, abs=0.001)

def test_Cn_yawrate_wing_two():
    """

    """

    ivc = om.IndepVarComp()
    # Flight reference condition
    ivc.add_output("data:reference_flight_condition:CL", val=0.1)
    ivc.add_output("data:reference_flight_condition:CD0", val=0.0186)

    # Wing Geometry
    x_c = 0.0
    aspect_ratio = 6.0
    sweep_25 = 40.0
    taper_ratio = 1.0
    ivc.add_output("data:geometry:wing:aspect_ratio", val=aspect_ratio)
    ivc.add_output("data:geometry:wing:taper_ratio", val=taper_ratio)
    ivc.add_output("data:geometry:wing:MAC:length", val=0.5, units="ft")
    ivc.add_output("data:geometry:wing:sweep_25", val=sweep_25, units="deg")

    ivc.add_output("data:handling_qualities:stick_fixed_static_margin", val=x_c)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(CnYawRateWing(), ivc)
    Cn_r_W = problem.get_val("data:handling_qualities:lateral:derivatives:wing:Cn:yawrate", units="rad**-1")
    assert Cn_r_W == pytest.approx(0.1452, abs=0.01)


