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
import openmdao.api as om
import numpy as np
import sympy as sym
from fastga.models.handling_qualities.utils.drag_compressibility import *


# TODO: register class
class LongitudinalMatrixSymPy(om.ExplicitComponent):
    # TODOC
    """

    """

    def setup(self):
        # Reference Flight Conditions
        self.add_input("data:reference_flight_condition:CL", val=np.nan)
        self.add_input("data:reference_flight_condition:CD", val=np.nan)
        self.add_input("data:reference_flight_condition:CT", val=np.nan)
        self.add_input("data:reference_flight_condition:air_density", val=np.nan, units="slug/ft**3")
        self.add_input("data:reference_flight_condition:theta", val=np.nan, units="deg")
        self.add_input("data:reference_flight_condition:weight", val=np.nan, units="slug")
        self.add_input("data:reference_flight_condition:speed", val=np.nan, units="ft/s")

        self.add_input("data:geometry:wing:MAC:length", val=np.nan, units="ft")
        self.add_input("data:geometry:wing:area", val=np.nan, units="ft**2")

        self.add_input("data:weight:aircraft:inertia:Ioy", val=np.nan, units="slug*ft**2")

        self.add_input("data:handling_qualities:longitudinal:derivatives:CL:speed", val=np.nan)
        self.add_input("data:handling_qualities:longitudinal:derivatives:CD:speed", val=np.nan)
        self.add_input("data:handling_qualities:longitudinal:derivatives:Cm:speed", val=np.nan)
        self.add_input("data:handling_qualities:longitudinal:derivatives:thrust:CX:speed", val=np.nan)
        self.add_input("data:handling_qualities:longitudinal:derivatives:thrust:Cm:speed", val=np.nan)

        self.add_input("data:handling_qualities:longitudinal:derivatives:CD:alpha", val=np.nan, units="rad**-1")
        self.add_input("data:handling_qualities:longitudinal:derivatives:CL:alpha", val=np.nan, units="rad**-1")
        self.add_input("data:handling_qualities:longitudinal:derivatives:Cm:alpha", val=np.nan, units="rad**-1")
        self.add_input("data:handling_qualities:longitudinal:derivatives:thrust:Cm:alpha", val=np.nan, units="rad**-1")

        self.add_input("data:handling_qualities:longitudinal:derivatives:CL:alpharate", val=np.nan, units="rad**-1")
        self.add_input("data:handling_qualities:longitudinal:derivatives:Cm:alpharate", val=np.nan, units="rad**-1")

        self.add_input("data:handling_qualities:longitudinal:derivatives:CL:pitchrate", val=np.nan, units="rad**-1")
        self.add_input("data:handling_qualities:longitudinal:derivatives:Cm:pitchrate", val=np.nan, units="rad**-1")


        # TODO: add outputs
        self.add_output("data:handling_qualities:longitudinal:spacestate:matrixA", units="rad**-1")
        # self.add_output("data:handling_qualities:longitudinal:spacestate:matrixB", units="rad**-1")
        self.add_output("data:handling_qualities:longitudinal:spacestate:eigenvalues")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        # Reference condition parameters:
        ro = inputs["data:reference_flight_condition:air_density"]  # or function of altitude
        theta_s = inputs["data:reference_flight_condition:theta"]
        CL_s = inputs["data:reference_flight_condition:CL"]
        CZ_s = - CL_s
        CD_s = inputs["data:reference_flight_condition:CD"]
        CT_s = inputs["data:reference_flight_condition:CT"]
        CX_s = CT_s - CD_s
        weight = inputs["data:reference_flight_condition:weight"]
        # NOTE: weight and mass are treated the same.
        mass = weight
        u_s = float(inputs["data:reference_flight_condition:speed"])
        q_s = 0.5 * ro * u_s**2

        # Geometry of wing, horizontal tail and propulsion system
        S = inputs["data:geometry:wing:area"]
        c = inputs["data:geometry:wing:MAC:length"]

        # Mass and moments of inertia parameters:
        # TODO: compute inertia parameters
        mu = mass / (ro * S * c / 2.0)
        Iyy = inputs["data:weight:aircraft:inertia:Ioy"]
        # Iyy = Iyy / (ro * S * (c / 2.0)**3)

        ###### SPEED DERIVATIVES ######
        # Aerodynamics
        CL_u = inputs["data:handling_qualities:longitudinal:derivatives:CL:speed"]
        CD_u = inputs["data:handling_qualities:longitudinal:derivatives:CD:speed"]
        CX_A_u = - CD_u
        CZ_A_u = - CL_u
        Cm_A_u = inputs["data:handling_qualities:longitudinal:derivatives:Cm:speed"]

        # Thrust
        CX_T_u = inputs["data:handling_qualities:longitudinal:derivatives:thrust:CX:speed"]
        # NOTE: we assume zero angle of attack for the engines.
        CZ_T_u = 0.0
        Cm_T_u = inputs["data:handling_qualities:longitudinal:derivatives:thrust:Cm:speed"]
        Cm_T_u = Cm_T_u

        CX_u = CX_A_u + CX_T_u
        CZ_u = CZ_A_u + CZ_T_u
        Cm_u = Cm_A_u + Cm_T_u

        ###### ALPHA DERIVATIVES ######
        CD_alpha = inputs["data:handling_qualities:longitudinal:derivatives:CD:alpha"]
        CL_alpha = inputs["data:handling_qualities:longitudinal:derivatives:CL:alpha"]
        CX_alpha = CL_s - CD_alpha
        CZ_alpha = - CL_alpha - CD_s
        Cm_alpha = inputs["data:handling_qualities:longitudinal:derivatives:Cm:alpha"]
        # Thrust
        # NOTE: for the moment the value of Cm_T_alpha = 0.0
        Cm_T_alpha = inputs["data:handling_qualities:longitudinal:derivatives:thrust:Cm:alpha"]
        Cm_alpha = Cm_alpha + Cm_T_alpha

        ###### ALPHA RATE DERIVATIVES ######
        CL_alpharate = inputs["data:handling_qualities:longitudinal:derivatives:CL:alpharate"]
        CD_alpharate = 0.0
        CZ_alpharate = - CL_alpharate
        Cm_alpharate = inputs["data:handling_qualities:longitudinal:derivatives:Cm:alpharate"]

        ###### PITCH RATE DERIVATIVES ######
        CL_q = inputs["data:handling_qualities:longitudinal:derivatives:CL:pitchrate"]
        CD_q = 0.0
        CZ_q = - CL_q
        Cm_q = inputs["data:handling_qualities:longitudinal:derivatives:Cm:pitchrate"]

        ###### DELTA ELEVATOR DERIVATIVES ######
        # TODO: compute deltae derivatives
        # CZ_deltae =
        # Cm_deltae =
        # CX_deltae =

        # Dimensional Derivatives (Roskam)
        X_u = float(q_s * S / u_s * (2*CX_s + CX_u) / mass)
        Z_u = float(q_s * S / u_s * (2*CZ_s + CZ_u) / mass)
        M_u = float(q_s * S / u_s * c * Cm_u / Iyy)
        X_a = float(q_s * S / u_s * CX_alpha * u_s/mass)
        Z_a = float(q_s * S / u_s * CZ_alpha * u_s/mass)
        M_a = float(q_s * S / u_s * c * Cm_alpha * u_s/Iyy)
        Z_a_dot = float(q_s * S * c / (2*u_s**2) * CZ_alpharate * u_s/mass)
        M_a_dot = float(q_s * S * c**2 / (2*u_s**2) * Cm_alpharate * u_s/Iyy)
        Z_q = float(q_s * S * c / (2*u_s) * CZ_q / mass)
        M_q = float(q_s * S * c**2 / (2*u_s) * Cm_q / Iyy)

        s = sym.Symbol('s')

        # Building the matrix (Roskam)
        a11 = s - X_u
        a12 = - X_a
        a13 = 9.81 * math.cos(theta_s)

        a21 = - Z_u
        a22 = s * (u_s - Z_a_dot) - Z_a
        a23 = - (Z_q + u_s) * s + 9.81 * math.sin(theta_s)

        a31 = - M_u
        a32 = - (M_a_dot * s + M_a)
        a33 = s**2 - M_q * s

        A = sym.Matrix([
            [a11, a12, a13],
            [a21, a22, a23],
            [a31, a32, a33]
        ])

        eq = A.det()
        sol = sym.solve(eq, s)
        sol_sp = sol[1]
        sol_ph = sol[3]

        ###### LONGITUDINAL MODES ######
        # Obtaining eigenvalues from matrix A
        # t_adim = c / (2*u_s)
        # w = w / t_adim

        # Selection of the eigenvalues of each mode
        sol_sp = sol[1]
        real_sp = sym.re(sol_sp)
        imag_sp = sym.im(sol_sp)
        wn_sp = math.sqrt(real_sp**2 + imag_sp**2)
        dump_sp = - real_sp / math.sqrt(real_sp**2 + imag_sp**2)
        period_sp = 2*math.pi / imag_sp

        # TODO: the result of the phugoid method are far from being close to the test data.
        sol_ph = sol[3]
        real_ph = sym.re(sol_ph)
        imag_ph = sym.im(sol_ph)
        wn_ph = math.sqrt(real_ph**2 + imag_ph**2)
        dump_ph = - real_ph / math.sqrt(real_ph**2 + imag_ph**2)
        period_ph = 2*math.pi / imag_ph

        outputs["data:handling_qualities:longitudinal:spacestate:matrixA"] = A
        # outputs["data:handling_qualities:longitudinal:spacestate:matrixB"] = B

        outputs["data:handling_qualities:longitudinal:spacestate:eigenvalues"] = w
