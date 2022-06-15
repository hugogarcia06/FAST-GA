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

import numpy as np
import math

from fastga.models.handling_qualities.utils.figure_digitization import FigureDigitization2


# TODO: register class
class ClYawRateWing(FigureDigitization2):

    def setup(self):
        # Flight reference condition
        self.add_input("data:reference_flight_condition:mach", val=np.nan)
        self.add_input("data:reference_flight_condition:CL", val=np.nan)
        self.add_input("data:reference_flight_condition:flaps_deflection", val=np.nan)

        # Wing Geometry
        self.add_input("data:geometry:wing:aspect_ratio", val=np.nan)
        self.add_input("data:geometry:wing:taper_ratio", val=np.nan)
        self.add_input("data:geometry:wing:dihedral", val=np.nan, units="rad")
        self.add_input("data:geometry:wing:twist", val=np.nan, units="deg")
        self.add_input("data:geometry:wing:sweep_25", val=np.nan, units="rad")
        # self.add_input("data:geometry:flap:flap_location", val=np.nan)

        self.add_input("data:aerodynamics:flaps:landing:CL", val=np.nan)
        self.add_input("data:aerodynamics:flaps:takeoff:CL", val=np.nan)

        # Wing Aerodynamics
        self.add_input("data:aerodynamics:wing:airfoil:CL_alpha", val=np.nan, units="rad**-1")

        self.add_output("data:handling_qualities:lateral:derivatives:wing:Cl:yawrate", units="rad**-1")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        # Flight reference condition
        mach = inputs["data:reference_flight_condition:mach"]
        CL_s = inputs["data:reference_flight_condition:CL"]
        CL_W = CL_s
        delta_flaps = inputs["data:reference_flight_condition:flaps_deflection"]

        # Wing geometry
        A_W = inputs["data:geometry:wing:aspect_ratio"]
        taper_ratio_W = inputs["data:geometry:wing:taper_ratio"]
        dihedral_W = inputs["data:geometry:wing:dihedral"]
        twist_W = inputs["data:geometry:wing:twist"]
        sweep_25_W = inputs["data:geometry:wing:sweep_25"]
        # flap_location = inputs["data:geometry:flap:flap_location"]

        # Wing Aerodynamics
        cl_alpha_w = inputs["data:aerodynamics:wing:airfoil:CL_alpha"]

        Clr_CL = self.compute_Clr_CL(A_W, taper_ratio_W, sweep_25_W, mach)
        # Equation 10.84
        deltaClr_dihedral_W = 0.083 * (math.pi * A_W * math.sin(sweep_25_W))/(A_W + 4*math.cos(sweep_25_W))
        # Obtained from figure digitization of Figure 10.42
        deltaClr_twist_W = self.get_delta_Clr_twist(A_W, taper_ratio_W)

        # Effect of symmetric flap deflection on rolling moment due to roll rate.
        # Obtained from figure digitization of Figure 10.43
        # TODO: introduce flap condition
        """if delta_flaps != np.nan:
            deltaClr_flaps = self.get_delta_Clr_flaps(flap_location, A_W, taper_ratio_W)
            # Equation 10.66 from Roskam
            delta_cl = np.nan
            if (landing condition):
                delta_cl = inputs["data:aerodynamics:flaps:landing:CL"]
            elif (takeoff condition):
                delta_cl = inputs["data:aerodynamics:flaps:takeoff:CL"]
            slope_flaps = delta_cl / (cl_alpha_w * delta_flaps)
        else:
            delta_flaps = 0.0
            slope_flaps = 0.0
            deltaClr_flaps = 0.0
        """
        deltaClr_flaps = 0.0
        slope_flaps = 0.0
        delta_flaps = 0.0
        # Equation 10.82 from Roskam. Important to take into account several things:
        # Twist angle must be in degrees (see Figure 10.42)
        # Dihedral angle must be in degrees
        # Flaps deflections must be in degrees (see Figure 10.43)
        Cl_r_W = Clr_CL * CL_W \
                 + deltaClr_dihedral_W * dihedral_W \
                 + deltaClr_twist_W * twist_W \
                 + deltaClr_flaps * slope_flaps * delta_flaps

        outputs["data:handling_qualities:lateral:derivatives:wing:Cl:yawrate"] = Cl_r_W

    def compute_Clr_CL(self,aspect_ratio, taper_ratio, sweep_25, mach):
        """
        Computation of the slope of the rolling moment due to roll rate at zero lift. The expression used is
        Equation 10.83 from Roskam - Airplane Design Part VI: Preliminary Calculation of Aerodynamic Thrust and
        Power Characteristics.

        :param aspect_ratio: surface aspect ratio.
        :param taper_ratio: surface taper ratio.
        :param sweep_25: quarter-chord point line sweep angle in radians.
        :param mach: reference flight condition mach number

        """
        A = aspect_ratio
        B = math.sqrt(1 - mach**2*(math.cos(sweep_25))**2)

        # This coefficient is obtained from figure digitization of Figure 10.41 from Roskam
        Clr_CL_mach0 = self.get_Clr_CL_mach0(aspect_ratio, taper_ratio, sweep_25)
        Clr_CL = Clr_CL_mach0 * (
                (
                1 +
                (A*(1-B**2)/(2*B*(A*B+2*math.cos(sweep_25)))) +
                ((math.tan(sweep_25))**2*(A*B+2*math.cos(sweep_25))/(A*B+4*math.cos(sweep_25))*8)
                )/(
                1 +
                ((math.tan(sweep_25))**2*(A+2*math.cos(sweep_25))/(A+4*math.cos(sweep_25))*8)
                )
        )

        return Clr_CL

