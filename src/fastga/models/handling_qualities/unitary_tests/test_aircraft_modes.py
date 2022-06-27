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

from fastga.models.handling_qualities.aircraft_modes_analysis import AircraftModesComputation
from fastga.models.handling_qualities.lateral_directional_dynamics.lateral_directional_spacestate import \
    LateralDirectionalSpaceStateMatrix

from fastga.models.handling_qualities.longitudinal_dynamics.longitudinal_modes import LongitudinalModes
from fastga.models.handling_qualities.longitudinal_dynamics.longitudinal_spacestate import LongitudinalSpaceStateMatrix
from fastga.models.handling_qualities.unitary_tests.test_functions import aircraft_modes
from tests.testing_utilities import run_system, get_indep_var_comp, list_inputs

def test_longitudinal_modes_one():
    """
    Data of Airplane A from Appendix B in Roskam - Airplane Flight Dynamics and Automatic Flight Controls Part I.
    """
    ivc = om.IndepVarComp()
    ivc.add_output("data:reference_flight_condition:CL", val=0.307)
    ivc.add_output("data:reference_flight_condition:CD", val=0.032)
    ivc.add_output("data:reference_flight_condition:CT", val=0.032)
    ivc.add_output("data:reference_flight_condition:air_density", val=0.00204834, units="slug/ft**3")
    ivc.add_output("data:reference_flight_condition:theta", val=0.0, units="deg")
    ivc.add_output("data:reference_flight_condition:weight", val=2650.0, units="lb")
    ivc.add_output("data:reference_flight_condition:speed", val=220.1, units="ft/s")

    ivc.add_output("data:geometry:wing:MAC:length", val=4.9, units="ft")
    ivc.add_output("data:geometry:wing:area", val=174.0, units="ft**2")

    ivc.add_output("data:weight:aircraft:inertia:Ioy", val=1346.0, units="slug*ft**2")

    ivc.add_output("data:handling_qualities:longitudinal:derivatives:CL:speed", val=0.0)
    ivc.add_output("data:handling_qualities:longitudinal:derivatives:CD:speed", val=0.0)
    ivc.add_output("data:handling_qualities:longitudinal:derivatives:Cm:speed", val=0.0)
    ivc.add_output("data:handling_qualities:longitudinal:derivatives:thrust:CX:speed", val=-0.096)
    ivc.add_output("data:handling_qualities:longitudinal:derivatives:thrust:Cm:speed", val=0.0)

    ivc.add_output("data:handling_qualities:longitudinal:derivatives:CD:alpha", val=0.121, units="rad**-1")
    ivc.add_output("data:handling_qualities:longitudinal:derivatives:CL:alpha", val=4.41, units="rad**-1")
    ivc.add_output("data:handling_qualities:longitudinal:derivatives:Cm:alpha", val=-0.613, units="rad**-1")
    ivc.add_output("data:handling_qualities:longitudinal:derivatives:thrust:Cm:alpha", val=0.0, units="rad**-1")

    ivc.add_output("data:handling_qualities:longitudinal:derivatives:CL:alpharate", val=1.7, units="rad**-1")
    ivc.add_output("data:handling_qualities:longitudinal:derivatives:Cm:alpharate", val=-7.27, units="rad**-1")

    ivc.add_output("data:handling_qualities:longitudinal:derivatives:CL:pitchrate", val=3.9, units="rad**-1")
    ivc.add_output("data:handling_qualities:longitudinal:derivatives:Cm:pitchrate", val=-12.4, units="rad**-1")

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(LongitudinalSpaceStateMatrix(), ivc)
    # problem = run_system(LongitudinalMatrixSymPy(), ivc)
    A = problem.get_val("data:handling_qualities:longitudinal:spacestate:matrixA", units="rad**-1")
    w = problem.get_val("data:handling_qualities:longitudinal:spacestate:eigenvalues")


    # assert CL_q_W == pytest.approx(0.129, abs=0.01)


def test_longitudinal_modes_two():
    """
    Data of Airplane E from Appendix B in Roskam - Airplane Flight Dynamics and Automatic Flight Controls Part I.
    """
    ivc = om.IndepVarComp()
    ivc.add_output("data:reference_flight_condition:CL", val=0.306)
    ivc.add_output("data:reference_flight_condition:CD", val=0.0298)
    ivc.add_output("data:reference_flight_condition:CT", val=0.0298)
    ivc.add_output("data:reference_flight_condition:air_density", val=0.00126659, units="slug/ft**3")
    ivc.add_output("data:reference_flight_condition:theta", val=1.1, units="deg")
    ivc.add_output("data:reference_flight_condition:weight", val=11000.0, units="lb")
    ivc.add_output("data:reference_flight_condition:speed", val=450.0, units="ft/s")

    ivc.add_output("data:geometry:wing:MAC:length", val=6.5, units="ft")
    ivc.add_output("data:geometry:wing:area", val=280.0, units="ft**2")

    ivc.add_output("data:weight:aircraft:inertia:Ioy", val=20250.0, units="slug*ft**2")

    ivc.add_output("data:handling_qualities:longitudinal:derivatives:CL:speed", val=0.02)
    ivc.add_output("data:handling_qualities:longitudinal:derivatives:CD:speed", val=0.0)
    ivc.add_output("data:handling_qualities:longitudinal:derivatives:Cm:speed", val=0.0)
    ivc.add_output("data:handling_qualities:longitudinal:derivatives:thrust:CX:speed", val=-0.0596)
    ivc.add_output("data:handling_qualities:longitudinal:derivatives:thrust:Cm:speed", val=0.0)

    ivc.add_output("data:handling_qualities:longitudinal:derivatives:CD:alpha", val=0.131, units="rad**-1")
    ivc.add_output("data:handling_qualities:longitudinal:derivatives:CL:alpha", val=5.48, units="rad**-1")
    ivc.add_output("data:handling_qualities:longitudinal:derivatives:Cm:alpha", val=-1.89, units="rad**-1")
    ivc.add_output("data:handling_qualities:longitudinal:derivatives:thrust:Cm:alpha", val=0.0, units="rad**-1")

    ivc.add_output("data:handling_qualities:longitudinal:derivatives:CL:alpharate", val=2.5, units="rad**-1")
    ivc.add_output("data:handling_qualities:longitudinal:derivatives:Cm:alpharate", val=-9.1, units="rad**-1")

    ivc.add_output("data:handling_qualities:longitudinal:derivatives:CL:pitchrate", val=8.1, units="rad**-1")
    ivc.add_output("data:handling_qualities:longitudinal:derivatives:Cm:pitchrate", val=-34.0, units="rad**-1")

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(LongitudinalSpaceStateMatrix(), ivc)
    A = problem.get_val("data:handling_qualities:longitudinal:spacestate:matrixA", units="rad**-1")
    w = problem.get_val("data:handling_qualities:longitudinal:spacestate:eigenvalues")


    # assert CL_q_W == pytest.approx(0.129, abs=0.01)


def test_longitudinal_modes_three():
    """
    Data of Airplane C from Appendix B in Roskam - Airplane Flight Dynamics and Automatic Flight Controls Part I.
    """
    ivc = om.IndepVarComp()
    ivc.add_output("data:reference_flight_condition:CL", val=1.414)
    ivc.add_output("data:reference_flight_condition:CD", val=0.210)
    ivc.add_output("data:reference_flight_condition:CT", val=0.210)
    ivc.add_output("data:reference_flight_condition:air_density", val=0.00237717, units="slug/ft**3")
    ivc.add_output("data:reference_flight_condition:theta", val=8.0, units="deg")
    ivc.add_output("data:reference_flight_condition:weight", val=3500.0, units="lb")
    ivc.add_output("data:reference_flight_condition:speed", val=73.46, units="kn")

    ivc.add_output("data:geometry:wing:MAC:length", val=5.4, units="ft")
    ivc.add_output("data:geometry:wing:area", val=136.0, units="ft**2")

    ivc.add_output("data:weight:aircraft:inertia:Ioy", val=4600.0, units="slug*ft**2")

    ivc.add_output("data:handling_qualities:longitudinal:derivatives:CL:speed", val=0.071)
    ivc.add_output("data:handling_qualities:longitudinal:derivatives:CD:speed", val=0.0)
    ivc.add_output("data:handling_qualities:longitudinal:derivatives:Cm:speed", val=0.0)
    ivc.add_output("data:handling_qualities:longitudinal:derivatives:thrust:CX:speed", val=-0.450)
    ivc.add_output("data:handling_qualities:longitudinal:derivatives:thrust:Cm:speed", val=0.0)

    ivc.add_output("data:handling_qualities:longitudinal:derivatives:CD:alpha", val=1.140, units="rad**-1")
    ivc.add_output("data:handling_qualities:longitudinal:derivatives:CL:alpha", val=5.0, units="rad**-1")
    ivc.add_output("data:handling_qualities:longitudinal:derivatives:Cm:alpha", val=-0.60, units="rad**-1")
    ivc.add_output("data:handling_qualities:longitudinal:derivatives:thrust:Cm:alpha", val=0.0, units="rad**-1")

    ivc.add_output("data:handling_qualities:longitudinal:derivatives:CL:alpharate", val=3.0, units="rad**-1")
    ivc.add_output("data:handling_qualities:longitudinal:derivatives:Cm:alpharate", val=-7.0, units="rad**-1")

    ivc.add_output("data:handling_qualities:longitudinal:derivatives:CL:pitchrate", val=9.0, units="rad**-1")
    ivc.add_output("data:handling_qualities:longitudinal:derivatives:Cm:pitchrate", val=-15.7, units="rad**-1")

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(LongitudinalSpaceStateMatrix(), ivc)
    A = problem.get_val("data:handling_qualities:longitudinal:spacestate:matrixA", units="rad**-1")
    w = problem.get_val("data:handling_qualities:longitudinal:spacestate:eigenvalues")


def test_longitudinal_modes_four():
    """
    Data from: https://courses.cit.cornell.edu/mae5070/DynamicStability.pdf
    """
    ivc = om.IndepVarComp()
    ivc.add_output("data:reference_flight_condition:CL", val=1.108)
    ivc.add_output("data:reference_flight_condition:CD", val=0.102)
    ivc.add_output("data:reference_flight_condition:CT", val=0.102)
    ivc.add_output("data:reference_flight_condition:air_density", val=0.002377, units="slug/ft**3")
    ivc.add_output("data:reference_flight_condition:theta", val=0.0, units="deg")
    ivc.add_output("data:reference_flight_condition:weight", val=564032.0, units="lb")
    ivc.add_output("data:reference_flight_condition:speed", val=279.1, units="ft/s")

    ivc.add_output("data:geometry:wing:MAC:length", val=27.3, units="ft")
    ivc.add_output("data:geometry:wing:area", val=5500, units="ft**2")

    ivc.add_output("data:weight:aircraft:inertia:Ioy", val=32300000.0, units="slug*ft**2")

    ivc.add_output("data:handling_qualities:longitudinal:derivatives:CL:speed", val=0.0)
    ivc.add_output("data:handling_qualities:longitudinal:derivatives:CD:speed", val=0.20371)
    ivc.add_output("data:handling_qualities:longitudinal:derivatives:Cm:speed", val=0.0)
    ivc.add_output("data:handling_qualities:longitudinal:derivatives:thrust:CX:speed", val=0.0)
    ivc.add_output("data:handling_qualities:longitudinal:derivatives:thrust:Cm:speed", val=0.0)

    ivc.add_output("data:handling_qualities:longitudinal:derivatives:CD:alpha", val=0.66, units="rad**-1")
    ivc.add_output("data:handling_qualities:longitudinal:derivatives:CL:alpha", val=5.7, units="rad**-1")
    ivc.add_output("data:handling_qualities:longitudinal:derivatives:Cm:alpha", val=-1.26, units="rad**-1")
    ivc.add_output("data:handling_qualities:longitudinal:derivatives:thrust:Cm:alpha", val=0.0, units="rad**-1")

    ivc.add_output("data:handling_qualities:longitudinal:derivatives:CL:alpharate", val=6.7, units="rad**-1")
    ivc.add_output("data:handling_qualities:longitudinal:derivatives:Cm:alpharate", val=-3.2, units="rad**-1")

    ivc.add_output("data:handling_qualities:longitudinal:derivatives:CL:pitchrate", val=5.4, units="rad**-1")
    ivc.add_output("data:handling_qualities:longitudinal:derivatives:Cm:pitchrate", val=-20.8, units="rad**-1")

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(LongitudinalSpaceStateMatrix(), ivc)
    A = problem.get_val("data:handling_qualities:longitudinal:spacestate:matrixA", units="rad**-1")
    w = problem.get_val("data:handling_qualities:longitudinal:spacestate:eigenvalues")


def test_lateral_directional_modes():
    """
    Data from Appendix B in Roskam - Airplane Flight Dynamics and Automatic Flight Controls Part I.
    """

    ivc = om.IndepVarComp()
    ivc.add_output("data:reference_flight_condition:CL", val=0.307)
    ivc.add_output("data:reference_flight_condition:air_density", val=1.05555, units="kg/m**3")
    ivc.add_output("data:reference_flight_condition:theta", val=0.0, units="deg")
    ivc.add_output("data:reference_flight_condition:weight", val=2650.0, units="lb")

    ivc.add_output("data:geometry:wing:span", val=36.0, units="ft")
    ivc.add_output("data:geometry:wing:area", val=174.0, units="ft**2")

    ivc.add_output("data:weight:aircraft:inertia:Iox", val=948.0, units="slug*ft**2")
    ivc.add_output("data:weight:aircraft:inertia:Ioz", val=1967, units="slug*ft**2")
    ivc.add_output("data:weight:aircraft:inertia:Ioxz", val=0.0, units="slug*ft**2")

    ivc.add_output("data:handling_qualities:lateral:derivatives:CY:beta", val=-0.393, units="rad**-1")
    ivc.add_output("data:handling_qualities:lateral:derivatives:CY:rollrate", val=-0.075, units="rad**-1")
    ivc.add_output("data:handling_qualities:lateral:derivatives:CY:yawrate", val=0.214, units="rad**-1")

    ivc.add_output("data:handling_qualities:lateral:derivatives:Cl:beta", val=-0.0923, units="rad**-1")
    ivc.add_output("data:handling_qualities:lateral:derivatives:Cl:rollrate", val=-0.484, units="rad**-1")
    ivc.add_output("data:handling_qualities:lateral:derivatives:Cl:yawrate", val=0.0798, units="rad**-1")

    ivc.add_output("data:handling_qualities:lateral:derivatives:Cn:beta", val=0.0587, units="rad**-1")
    ivc.add_output("data:handling_qualities:lateral:derivatives:Cn:rollrate", val=-0.0278, units="rad**-1")
    ivc.add_output("data:handling_qualities:lateral:derivatives:Cn:yawrate", val=-0.0937, units="rad**-1")

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(LateralDirectionalSpaceStateMatrix(), ivc)
    A = problem.get_val("data:handling_qualities:lateral:spacestate:matrixA", units="rad**-1")
    w = problem.get_val("data:handling_qualities:lateral:spacestate:eigenvalues")


def test_lateral_directional_modes_two():
    """
    Data from: https://courses.cit.cornell.edu/mae5070/DynamicStability.pdf
    """

    ivc = om.IndepVarComp()
    ivc.add_output("data:reference_flight_condition:CL", val=1.108)
    ivc.add_output("data:reference_flight_condition:air_density", val=0.002377, units="slug/ft**3")
    ivc.add_output("data:reference_flight_condition:theta", val=0.0, units="deg")
    ivc.add_output("data:reference_flight_condition:weight", val=564032.0, units="lb")
    ivc.add_output("data:reference_flight_condition:speed", val=279.1, units="ft/s")

    ivc.add_output("data:geometry:wing:span", val=195.7, units="ft")
    ivc.add_output("data:geometry:wing:area", val=5500.0, units="ft**2")

    ivc.add_output("data:weight:aircraft:inertia:Iox", val=14300000.0, units="slug*ft**2")
    ivc.add_output("data:weight:aircraft:inertia:Ioz", val=45300000, units="slug*ft**2")
    ivc.add_output("data:weight:aircraft:inertia:Ioxz", val=-2230000, units="slug*ft**2")

    ivc.add_output("data:handling_qualities:lateral:derivatives:CY:beta", val=-0.96, units="rad**-1")
    ivc.add_output("data:handling_qualities:lateral:derivatives:CY:rollrate", val=0.0, units="rad**-1")
    ivc.add_output("data:handling_qualities:lateral:derivatives:CY:yawrate", val=0.0, units="rad**-1")

    ivc.add_output("data:handling_qualities:lateral:derivatives:Cl:beta", val=-0.221, units="rad**-1")
    ivc.add_output("data:handling_qualities:lateral:derivatives:Cl:rollrate", val=-0.45, units="rad**-1")
    ivc.add_output("data:handling_qualities:lateral:derivatives:Cl:yawrate", val=0.101, units="rad**-1")

    ivc.add_output("data:handling_qualities:lateral:derivatives:Cn:beta", val=0.15, units="rad**-1")
    ivc.add_output("data:handling_qualities:lateral:derivatives:Cn:rollrate", val=-0.121, units="rad**-1")
    ivc.add_output("data:handling_qualities:lateral:derivatives:Cn:yawrate", val=-0.30, units="rad**-1")

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(LateralDirectionalSpaceStateMatrix(), ivc)
    A = problem.get_val("data:handling_qualities:lateral:spacestate:matrixA", units="rad**-1")
    w = problem.get_val("data:handling_qualities:lateral:spacestate:eigenvalues")


def test_aircraft_modes():
    """
    For testing the complete modes analysis, from calculating the stability derivatives until obtaining the modes
    characteristics.
    """

    use_openvsp = True
    add_fuselage = False
    XML_FILE = "beechcraft_76.xml"

    aircraft_modes(
        use_openvsp,
        add_fuselage,
        XML_FILE
    )