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
import pytest

from fastga.models.weight.inertia.a_airframe.fuselage_inertia import ComputeFuselageInertia
from fastga.models.weight.inertia.a_airframe.horizontal_tail_inertia import ComputeHorizontalTailInertia
from fastga.models.weight.inertia.a_airframe.vertical_tail_inertia import ComputeVerticalTailInertia
from fastga.models.weight.inertia.a_airframe.wing_inertia import ComputeWingInertia
from fastga.models.weight.inertia.aircraft_inertia import ComputeAircraftInertia

from fastga.models.weight.inertia.b_propulsion.power_plant_inertia import ComputePowerPlantInertia
from fastga.models.weight.inertia.b_propulsion.fuel_inertia import ComputeFuelInertia
from tests.testing_utilities import run_system


def test_inertia_0_wing():
    """
    Test data from USAF DATCOM (pdf), section 8.1, page 2835/3070.
    """


    ivc = om.IndepVarComp()
    ivc.add_output("data:weight:airframe:wing:mass", val=15000, units="lb")
    ivc.add_output("data:geometry:wing:root:chord", val=300, units="inch")
    ivc.add_output("data:geometry:wing:tip:chord", val=100, units="inch")
    ivc.add_output("data:geometry:wing:span", val=1000, units="inch")
    ivc.add_output("data:geometry:wing:sweep_0", val=12.1, units="deg")

    ivc.add_output("data:weight:airframe:wing:CG:x", val=650.0, units="inch")
    ivc.add_output("data:weight:airframe:half_wing:CG:y", val=150.0, units="inch")


    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputeWingInertia(), ivc)
    Io_x = problem.get_val("data:weight:airframe:wing:inertia:Iox", units="lb*inch**2")
    Io_y = problem.get_val("data:weight:airframe:wing:inertia:Ioy", units="lb*inch**2")
    Io_z = problem.get_val("data:weight:airframe:wing:inertia:Ioz", units="lb*inch**2")
    assert Io_x == pytest.approx(628125000, rel=0.01)
    assert Io_y == pytest.approx(43976971, rel=0.01)
    assert Io_z == pytest.approx(672101971, rel=0.01)

    W_x = problem.get_val("data:weight:airframe:wing:inertia:Wx", units="lb*inch")
    W_y = problem.get_val("data:weight:airframe:wing:inertia:Wy", units="lb*inch")
    W_z = problem.get_val("data:weight:airframe:wing:inertia:Wz", units="lb*inch")
    assert W_x == pytest.approx(9750000, rel=0.01)
    assert W_y == pytest.approx(0.0, abs=0.01)
    assert W_z == pytest.approx(2250000, rel=0.01)


def test_inertia_0_ht():
    """
    Test data from USAF DATCOM (pdf), section 8.1, page 2835/3070.
    """


    ivc = om.IndepVarComp()
    ivc.add_output("data:weight:airframe:horizontal_tail:mass", val=1000, units="lb")
    ivc.add_output("data:geometry:horizontal_tail:root:chord", val=100.0, units="inch")
    ivc.add_output("data:geometry:horizontal_tail:tip:chord", val=50.0, units="inch")
    ivc.add_output("data:geometry:horizontal_tail:span", val=400.0, units="inch")
    ivc.add_output("data:geometry:horizontal_tail:sweep_0", val=12.0, units="deg")
    ivc.add_output("data:weight:airframe:half_horizontal_tail:CG:y", val=80, units="inch")


    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputeHorizontalTailInertia(), ivc)
    Io_x = problem.get_val("data:weight:airframe:horizontal_tail:inertia:Iox", units="lb*inch**2")
    Io_y = problem.get_val("data:weight:airframe:horizontal_tail:inertia:Ioy", units="lb*inch**2")
    Io_z = problem.get_val("data:weight:airframe:horizontal_tail:inertia:Ioz", units="lb*inch**2")
    assert Io_x == pytest.approx(8222222, rel=0.05)
    assert Io_y == pytest.approx(443070, rel=0.06)
    assert Io_z == pytest.approx(8665292, rel=0.05)


def test_inertia_0_vt():
    """
    Test data from USAF DATCOM (pdf), section 8.1, page 2835/3070.
    """


    ivc = om.IndepVarComp()
    ivc.add_output("data:weight:airframe:vertical_tail:mass", val=300.0, units="lb")
    ivc.add_output("data:geometry:vertical_tail:root:chord", val=250.0, units="inch")
    ivc.add_output("data:geometry:vertical_tail:tip:chord", val=100.0, units="inch")
    ivc.add_output("data:geometry:vertical_tail:span", val=200.0, units="inch")
    ivc.add_output("data:geometry:vertical_tail:sweep_0", val=37.0, units="deg")
    ivc.add_output("data:weight:airframe:vertical_tail:CG:z", val=75.0, units="inch")


    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputeVerticalTailInertia(), ivc)
    Io_x = problem.get_val("data:weight:airframe:vertical_tail:inertia:Iox", units="lb*inch**2")
    Io_y = problem.get_val("data:weight:airframe:vertical_tail:inertia:Ioy", units="lb*inch**2")
    Io_z = problem.get_val("data:weight:airframe:vertical_tail:inertia:Ioz", units="lb*inch**2")
    assert Io_x == pytest.approx(872960, rel=0.07)
    assert Io_y == pytest.approx(1769402, rel=0.07)
    assert Io_z == pytest.approx(896442, rel=0.11)
    
    
def test_inertia_0_fuselage():
    """
    Test data from USAF DATCOM (pdf), section 8.1, page 2835/3070.
    """


    ivc = om.IndepVarComp()
    ivc.add_output("data:weight:airframe:fuselage:mass", val=20000.0, units="lb")
    ivc.add_output("data:geometry:fuselage:wet_area", val=400000.0, units="inch**2")
    ivc.add_output("data:geometry:fuselage:maximum_height", val=150.0, units="inch")
    ivc.add_output("data:geometry:fuselage:maximum_width", val=150.0, units="inch")
    ivc.add_output("data:geometry:fuselage:length", val=1200.0, units="inch")
    ivc.add_output("data:weight:airframe:fuselage:CG:x", val=500.0, units="inch")

    ivc.add_output("data:weight:airframe:fuselage:shell:mass", val=2000.0, units="lb")
    ivc.add_output("data:weight:airframe:fuselage:bulkhead:mass", val=6000.0, units="lb")

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputeFuselageInertia(), ivc)
    Io_x = problem.get_val("data:weight:airframe:fuselage:inertia:Iox", units="lb*inch**2")
    Io_y = problem.get_val("data:weight:airframe:fuselage:inertia:Ioy", units="lb*inch**2")
    Io_z = problem.get_val("data:weight:airframe:fuselage:inertia:Ioz", units="lb*inch**2")
    assert Io_x == pytest.approx(54600860, rel=0.01)
    assert Io_y == pytest.approx(1442807855, rel=0.05)
    assert Io_z == pytest.approx(1422807855, rel=0.05)


def test_inertia_0_powerplant():
    """
    Test data from USAF DATCOM (pdf), section 8.1, page 2835/3070.
    """

    ivc = om.IndepVarComp()
    ivc.add_output("data:weight:propulsion:mass", val=10000.0, units="lb")
    ivc.add_output("data:geometry:propulsion:nacelle:length", val=200.0, units="inch")
    ivc.add_output("data:geometry:propulsion:nacelle:height", val=62.5, units="inch")
    ivc.add_output("data:geometry:propulsion:nacelle:width", val=62.5, units="inch")
    ivc.add_output("data:weight:propulsion:engine:mass", val=7000.0, units="lb")

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputePowerPlantInertia(), ivc)
    Io_x = problem.get_val("data:weight:propulsion:powerplant:inertia:Iox", units="lb*inch**2")
    Io_y = problem.get_val("data:weight:propulsion:powerplant:inertia:Ioy", units="lb*inch**2")
    Io_z = problem.get_val("data:weight:propulsion:powerplant:inertia:Ioz", units="lb*inch**2")
    assert Io_x == pytest.approx(2075000, rel=0.01)
    assert Io_y == pytest.approx(12733750, rel=0.01)
    assert Io_z == pytest.approx(12733750, rel=0.01)


def test_inertia_0_fuel():
    """
    Test data from USAF DATCOM (pdf), section 8.1, page 2839/3070.
    """

    ivc = om.IndepVarComp()
    ivc.add_output("data:reference_flight_condition:fuel_weight", val=20000.0, units="lb")
    ivc.add_output("data:geometry:wing:root:chord", val=150.0, units="inch")
    ivc.add_output("data:geometry:wing:root:thickness_ratio", val=0.05333)
    ivc.add_output("data:geometry:wing:tip:chord", val=150.0, units="inch")
    ivc.add_output("data:geometry:wing:tip:thickness_ratio", val=0.05333)
    ivc.add_output("data:geometry:wing:span", val=600.0, units="inch")

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputeFuelInertia(), ivc)
    Io_x = problem.get_val("data:weight:propulsion:fuel:inertia:Iox", units="lb*inch**2")
    Io_y = problem.get_val("data:weight:propulsion:fuel:inertia:Ioy", units="lb*inch**2")
    Io_z = problem.get_val("data:weight:propulsion:fuel:inertia:Ioz", units="lb*inch**2")
    assert Io_x == pytest.approx(600106667, rel=0.01)
    assert Io_y == pytest.approx(37606667, rel=0.01)
    assert Io_z == pytest.approx(637500000, rel=0.01)


def test_inertia_aircraft():
    """
    Test data from USAF DATCOM (pdf), section 8.1, page 2839/3070.
    """

    ivc = om.IndepVarComp()

    # Wing
    ivc.add_output("data:weight:airframe:wing:inertia:Wx", val=9750000.0, units="lb*inch")
    ivc.add_output("data:weight:airframe:wing:inertia:Wy", val=0.0, units="lb*inch")
    ivc.add_output("data:weight:airframe:wing:inertia:Wz", val=2250000.0, units="lb*inch")

    ivc.add_output("data:weight:airframe:wing:inertia:Ix", val=965625000.0, units="lb*inch**2")
    ivc.add_output("data:weight:airframe:wing:inertia:Iy", val=6718977000.0, units="lb*inch**2")
    ivc.add_output("data:weight:airframe:wing:inertia:Iz", val=7009602000.0, units="lb*inch**2")
    ivc.add_output("data:weight:airframe:wing:inertia:Ixz", val=0.0, units="lb*inch**2")

    # Horizontal tail 
    ivc.add_output("data:weight:airframe:horizontal_tail:inertia:Wx", val=1150000.0, units="lb*inch")
    ivc.add_output("data:weight:airframe:horizontal_tail:inertia:Wy", val=0.0, units="lb*inch")
    ivc.add_output("data:weight:airframe:horizontal_tail:inertia:Wz", val=200000.0, units="lb*inch")

    ivc.add_output("data:weight:airframe:horizontal_tail:inertia:Ix", val=48222000.0, units="lb*inch**2")
    ivc.add_output("data:weight:airframe:horizontal_tail:inertia:Iy", val=1362943000.0, units="lb*inch**2")
    ivc.add_output("data:weight:airframe:horizontal_tail:inertia:Iz", val=1331165000.0, units="lb*inch**2")
    ivc.add_output("data:weight:airframe:horizontal_tail:inertia:Ixz", val=0.0, units="lb*inch**2")

    # Vertical tail
    ivc.add_output("data:weight:airframe:vertical_tail:inertia:Wx", val=360000.0, units="lb*inch")
    ivc.add_output("data:weight:airframe:vertical_tail:inertia:Wy", val=0.0, units="lb*inch")
    ivc.add_output("data:weight:airframe:vertical_tail:inertia:Wz", val=90000.0, units="lb*inch")

    ivc.add_output("data:weight:airframe:vertical_tail:inertia:Ix", val=27873000.0, units="lb*inch**2")
    ivc.add_output("data:weight:airframe:vertical_tail:inertia:Iy", val=460769000.0, units="lb*inch**2")
    ivc.add_output("data:weight:airframe:vertical_tail:inertia:Iz", val=432896000.0, units="lb*inch**2")
    ivc.add_output("data:weight:airframe:vertical_tail:inertia:Ixz", val=0.0, units="lb*inch**2")

    # Fuselage
    ivc.add_output("data:weight:airframe:fuselage:inertia:Wx", val=12000000.0, units="lb*inch")
    ivc.add_output("data:weight:airframe:fuselage:inertia:Wy", val=0.0, units="lb*inch")
    ivc.add_output("data:weight:airframe:fuselage:inertia:Wz", val=4000000.0, units="lb*inch")

    ivc.add_output("data:weight:airframe:fuselage:inertia:Ix", val=854601000.0, units="lb*inch**2")
    ivc.add_output("data:weight:airframe:fuselage:inertia:Iy", val=9442808000.0, units="lb*inch**2")
    ivc.add_output("data:weight:airframe:fuselage:inertia:Iz", val=8642808000.0, units="lb*inch**2")
    ivc.add_output("data:weight:airframe:fuselage:inertia:Ixz", val=0.0, units="lb*inch**2")

    # Landing Gear
    ivc.add_output("data:weight:airframe:landing_gear:front:inertia:Wx", val=0.0, units="lb*inch")
    ivc.add_output("data:weight:airframe:landing_gear:front:inertia:Wy", val=0.0, units="lb*inch")
    ivc.add_output("data:weight:airframe:landing_gear:front:inertia:Wz", val=0.0, units="lb*inch")

    ivc.add_output("data:weight:airframe:landing_gear:front:inertia:Ix", val=0.0, units="lb*inch**2")
    ivc.add_output("data:weight:airframe:landing_gear:front:inertia:Iy", val=0.0, units="lb*inch**2")
    ivc.add_output("data:weight:airframe:landing_gear:front:inertia:Iz", val=0.0, units="lb*inch**2")
    ivc.add_output("data:weight:airframe:landing_gear:front:inertia:Ixz", val=0.0, units="lb*inch**2")

    ivc.add_output("data:weight:airframe:landing_gear:main:inertia:Wx", val=0.0, units="lb*inch")
    ivc.add_output("data:weight:airframe:landing_gear:main:inertia:Wy", val=0.0, units="lb*inch")
    ivc.add_output("data:weight:airframe:landing_gear:main:inertia:Wz", val=0.0, units="lb*inch")

    ivc.add_output("data:weight:airframe:landing_gear:main:inertia:Ix", val=0.0, units="lb*inch**2")
    ivc.add_output("data:weight:airframe:landing_gear:main:inertia:Iy", val=0.0, units="lb*inch**2")
    ivc.add_output("data:weight:airframe:landing_gear:main:inertia:Iz", val=0.0, units="lb*inch**2")
    ivc.add_output("data:weight:airframe:landing_gear:main:inertia:Ixz", val=0.0, units="lb*inch**2")

    # Power plant 
    ivc.add_output("data:weight:propulsion:powerplant:inertia:Wx", val=5200000.0, units="lb*inch")
    ivc.add_output("data:weight:propulsion:powerplant:inertia:Wy", val=2000000.0, units="lb*inch")
    ivc.add_output("data:weight:propulsion:powerplant:inertia:Wz", val=1500000.0, units="lb*inch")

    ivc.add_output("data:weight:propulsion:powerplant:inertia:Ix", val=627075000.0, units="lb*inch**2")
    ivc.add_output("data:weight:propulsion:powerplant:inertia:Iy", val=2941734000.0, units="lb*inch**2")
    ivc.add_output("data:weight:propulsion:powerplant:inertia:Iz", val=3116734000.0, units="lb*inch**2")
    ivc.add_output("data:weight:propulsion:powerplant:inertia:Ixz", val=0.0, units="lb*inch**2")

    # Fuel
    ivc.add_output("data:weight:propulsion:fuel:inertia:Wx", val=5200000.0, units="lb*inch")
    ivc.add_output("data:weight:propulsion:fuel:inertia:Wy", val=2000000.0, units="lb*inch")
    ivc.add_output("data:weight:propulsion:fuel:inertia:Wz", val=1500000.0, units="lb*inch")

    ivc.add_output("data:weight:propulsion:fuel:inertia:Ix", val=1050107000.0, units="lb*inch**2")
    ivc.add_output("data:weight:propulsion:fuel:inertia:Iy", val=8937607000.0, units="lb*inch**2")
    ivc.add_output("data:weight:propulsion:fuel:inertia:Iz", val=9087500000.0, units="lb*inch**2")
    ivc.add_output("data:weight:propulsion:fuel:inertia:Ixz", val=0.0, units="lb*inch**2")

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputeAircraftInertia(), ivc)

    I_x = problem.get_val("data:weight:aircraft:inertia:Ix", units="lb*inch**2")
    I_y = problem.get_val("data:weight:aircraft:inertia:Iy", units="lb*inch**2")
    I_z = problem.get_val("data:weight:aircraft:inertia:Iz", units="lb*inch**2")
    assert I_x == pytest.approx(3573503000.0, rel=0.01)
    assert I_y == pytest.approx(29864838000.0, rel=0.01)
    assert I_z == pytest.approx(29620705000.0, rel=0.01)

    Io_x = problem.get_val("data:weight:aircraft:inertia:Iox", units="lb*inch**2")
    Io_y = problem.get_val("data:weight:aircraft:inertia:Ioy", units="lb*inch**2")
    Io_z = problem.get_val("data:weight:aircraft:inertia:Ioz", units="lb*inch**2")
    assert Io_x == pytest.approx(600106667, rel=0.01)
    assert Io_y == pytest.approx(37606667, rel=0.01)
    assert Io_z == pytest.approx(637500000, rel=0.01)


