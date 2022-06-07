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

from openmdao.core.explicitcomponent import ExplicitComponent


class LateralDirectionalSpaceStateMatrix(ExplicitComponent):
    # TODOC
    """
    Dimensionless Lateral-Directional Equations of motion in State-Space form.
    DX = AX + BU where X = {beta, p, r, phi}
    Y = CX + DU
    """
    def setup(self):

        self.add_input("data:reference_flight_condition:CL", val=np.nan)
        self.add_input("data:reference_flight_condition:air_density", val=np.nan, units="kg/m**3")
        self.add_input("data:reference_flight_condition:theta", val=np.nan, units="rad")
        self.add_input("data:reference_flight_condition:weight", val=np.nan, units="kg")

        self.add_input("data:geometry:wing:area", val=np.nan, units="m**2")
        self.add_input("data:geometry:wing:span", val=np.nan, units="m")

        self.add_input("data:weight:aircraft:inertia:Iox", val=np.nan, units="kg*m**2")
        self.add_input("data:weight:aircraft:inertia:Ioz", val=np.nan, units="kg*m**2")
        self.add_input("data:weight:aircraft:inertia:Ioxz", val=np.nan, units="kg*m**2")

        self.add_input("data:handling_qualities:lateral:derivatives:CY:beta", val=np.nan, units="rad**-1")
        self.add_input("data:handling_qualities:lateral:derivatives:CY:rollrate", val=np.nan, units="rad**-1")
        self.add_input("data:handling_qualities:lateral:derivatives:CY:yawrate", val=np.nan, units="rad**-1")

        self.add_input("data:handling_qualities:lateral:derivatives:Cl:beta", val=np.nan, units="rad**-1")
        self.add_input("data:handling_qualities:lateral:derivatives:Cl:rollrate", val=np.nan, units="rad**-1")
        self.add_input("data:handling_qualities:lateral:derivatives:Cl:yawrate", val=np.nan, units="rad**-1")

        self.add_input("data:handling_qualities:lateral:derivatives:Cn:beta", val=np.nan, units="rad**-1")
        self.add_input("data:handling_qualities:lateral:derivatives:Cn:rollrate", val=np.nan, units="rad**-1")
        self.add_input("data:handling_qualities:lateral:derivatives:Cn:yawrate", val=np.nan, units="rad**-1")

        self.add_output("data:handling_qualities:lateral:spacestate:matrixA", val=np.nan, units="rad**-1")
        self.add_output("data:handling_qualities:lateral:spacestate:eigenvalues")


    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        ro = inputs["data:reference_flight_condition:air_density"]  # or function of altitude
        CL_s = inputs["data:reference_flight_condition:CL"]
        CZ_s = - CL_s
        theta_s = inputs["data:reference_flight_condition:theta"]
        weight = inputs["data:reference_flight_condition:weight"]
        # NOTE: weight and mass are treated the same.
        mass = weight
        S = inputs["data:geometry:wing:area"]
        b = inputs["data:geometry:wing:span"]
        mu = 2 * mass / (ro * S * b / 2)

        # TODO: moments and products of inertia computation
        Ix = inputs["data:weight:aircraft:inertia:Iox"]
        Ix = Ix / (ro * S * (b / 2)**3)
        Iz = inputs["data:weight:aircraft:inertia:Ioz"]
        Iz = Iz / (ro * S * (b / 2)**3)
        Jxz = inputs["data:weight:aircraft:inertia:Ioxz"]
        Jxz = Jxz / (ro * S * (b / 2)**3)

        CY_beta = inputs["data:handling_qualities:lateral:derivatives:CY:beta"]
        CY_p = inputs["data:handling_qualities:lateral:derivatives:CY:rollrate"]
        CY_r = inputs["data:handling_qualities:lateral:derivatives:CY:yawrate"]

        Cl_beta = inputs["data:handling_qualities:lateral:derivatives:Cl:beta"]
        Cl_p = inputs["data:handling_qualities:lateral:derivatives:Cl:rollrate"]
        Cl_r = inputs["data:handling_qualities:lateral:derivatives:Cl:yawrate"]

        Cn_beta = inputs["data:handling_qualities:lateral:derivatives:Cn:beta"]
        Cn_p = inputs["data:handling_qualities:lateral:derivatives:Cn:rollrate"]
        Cn_r = inputs["data:handling_qualities:lateral:derivatives:Cn:yawrate"]


        a11 = float(CY_beta / mu)
        a12 = float(CY_p / (2*mu))
        a13 = float(- (2*mu - CY_r)/(2*mu))
        a14 = float(- CZ_s / (2*mu))
        a21 = float((Cl_beta*Iz + Cn_beta*Jxz) / (Ix*Iz - Jxz**2))
        a22 = float((Cl_p*Iz + Cn_p*Jxz) / (Ix*Iz - Jxz**2))
        a23 = float((Cl_r*Iz + Cn_r*Jxz) / (Ix*Iz - Jxz**2))
        a24 = 0.0
        a31 = float((Cn_beta*Ix + Cl_beta*Jxz) / (Ix*Iz - Jxz**2))
        a32 = float((Cn_p*Ix + Cl_p*Jxz) / (Ix*Iz - Jxz**2))
        a33 = float((Cn_r*Ix + Cl_r*Jxz) / (Ix*Iz - Jxz**2))
        a34 = 0.0
        a41 = 0.0
        a42 = 1.0
        a43 = float()
        a44 = math.tan(theta_s)

        A = np.array([[a11, a12, a13, a14],
                      [a21, a22, a23, a24],
                      [a31, a32, a33, a34],
                      [a41, a42, a43, a44]])


        ###### LATERAL-DIRECTIONAL MODES ######
        # Obtaining eigenvalues from matrix A
        w, v = np.linalg.eig(A)
        w.sort()

        outputs["data:handling_qualities:lateral:spacestate:matrixA"] = A

        outputs["data:handling_qualities:lateral:spacestate:eigenvalues"] = w


