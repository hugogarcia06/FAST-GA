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
import math
import numpy as np

import openmdao.api as om
from fastoad.model_base import Atmosphere


class ReferenceFlightCondition(om.ExplicitComponent):
    """ Establishes the reference flight condition for the stability analysis  """

    def setup(self):

        self.add_input("data:weight:aircraft:MTOW", val=np.nan, units="kg")
        self.add_input("data:geometry:wing:area", val=np.nan, units="m**2")
        self.add_input("data:aerodynamics:aircraft:cruise:CD0", val=np.nan)
        self.add_input("data:aerodynamics:wing:cruise:induced_drag_coefficient", val=np.nan)

        self.add_output("data:reference_flight_condition:mach")
        self.add_output("data:reference_flight_condition:alpha", units="deg")
        self.add_output("data:reference_flight_condition:theta", units="deg")
        self.add_output("data:reference_flight_condition:speed", units="m/s")
        self.add_output("data:reference_flight_condition:altitude", units="m")
        self.add_output("data:reference_flight_condition:weight", units="kg")
        self.add_output("data:reference_flight_condition:CL")
        self.add_output("data:reference_flight_condition:CD")
        self.add_output("data:reference_flight_condition:CD0")
        self.add_output("data:reference_flight_condition:CT")
        self.add_output("data:reference_flight_condition:dynamic_pressure", units="Pa")
        #self.add_output("data:reference_flight_condition:CG:x", units="m")
        self.add_output("data:reference_flight_condition:CG:z", units="m")
        self.add_output("data:reference_flight_condition:flaps_deflection", units="deg")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        S = inputs["data:geometry:wing:area"]

        mach = 0.2
        alpha = 0.0
        theta = 0.0
        altitude = 3000.0    # altitude in meters
        sound_speed = Atmosphere(altitude, False).speed_of_sound
        speed = mach * sound_speed
        rho = Atmosphere(altitude, False).density
        q = 0.5 * rho * speed**2

        weight = inputs["data:weight:aircraft:MTOW"] * 0.8
        L = math.cos(theta * math.pi / 180.0) * weight
        CL = L / (q * S)

        CD0 = inputs["data:aerodynamics:aircraft:cruise:CD0"]
        k = inputs["data:aerodynamics:wing:cruise:induced_drag_coefficient"]
        CD = CD0 + k * CL ** 2
        CT = CD + math.sin(theta * math.pi / 180.0) * weight / (q * S)

        flaps_deflection = 0.0

        cg_z = 0.0

        outputs["data:reference_flight_condition:mach"] = mach
        outputs["data:reference_flight_condition:alpha"] = alpha
        outputs["data:reference_flight_condition:theta"] = theta
        outputs["data:reference_flight_condition:speed"] = speed
        outputs["data:reference_flight_condition:weight"] = weight
        outputs["data:reference_flight_condition:CL"] = CL
        outputs["data:reference_flight_condition:CD"] = CD
        outputs["data:reference_flight_condition:CT"] = CT
        outputs["data:reference_flight_condition:dynamic_pressure"] = q
        #outputs["data:reference_flight_condition:CG:x"] = cg_x
        outputs["data:reference_flight_condition:CG:z"] = cg_z
        outputs["data:reference_flight_condition:flaps_deflection"] = flaps_deflection
        outputs["data:reference_flight_condition:CD0"] = CD0
