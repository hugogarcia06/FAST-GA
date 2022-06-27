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
    Dimensional Lateral-Directional Equations of motion in State-Space form.
    DX = AX + BU where X = {beta, p, r, phi}
    Y = CX + DU
    """
    def setup(self):

        self.add_input("data:reference_flight_condition:CL", val=np.nan)
        self.add_input("data:reference_flight_condition:air_density", val=np.nan, units="slug/ft**3")
        self.add_input("data:reference_flight_condition:theta", val=np.nan, units="deg")
        self.add_input("data:reference_flight_condition:weight", val=np.nan, units="slug")
        self.add_input("data:reference_flight_condition:speed", val=np.nan, units="ft/s")

        self.add_input("data:geometry:wing:area", val=np.nan, units="ft**2")
        self.add_input("data:geometry:wing:span", val=np.nan, units="ft")

        self.add_input("data:weight:aircraft:inertia:Iox", val=np.nan, units="slug*ft**2")
        self.add_input("data:weight:aircraft:inertia:Ioz", val=np.nan, units="slug*ft**2")
        self.add_input("data:weight:aircraft:inertia:Ioxz", val=np.nan, units="slug*ft**2")

        self.add_input("data:handling_qualities:lateral:derivatives:CY:beta", val=np.nan, units="rad**-1")
        self.add_input("data:handling_qualities:lateral:derivatives:CY:rollrate", val=np.nan, units="rad**-1")
        self.add_input("data:handling_qualities:lateral:derivatives:CY:yawrate", val=np.nan, units="rad**-1")

        self.add_input("data:handling_qualities:lateral:derivatives:Cl:beta", val=np.nan, units="rad**-1")
        self.add_input("data:handling_qualities:lateral:derivatives:Cl:rollrate", val=np.nan, units="rad**-1")
        self.add_input("data:handling_qualities:lateral:derivatives:Cl:yawrate", val=np.nan, units="rad**-1")

        self.add_input("data:handling_qualities:lateral:derivatives:Cn:beta", val=np.nan, units="rad**-1")
        self.add_input("data:handling_qualities:lateral:derivatives:Cn:rollrate", val=np.nan, units="rad**-1")
        self.add_input("data:handling_qualities:lateral:derivatives:Cn:yawrate", val=np.nan, units="rad**-1")

        self.add_output("data:handling_qualities:lateral:modes:dutch_roll:real_part", units="s**-1")
        self.add_output("data:handling_qualities:lateral:modes:dutch_roll:imag_part", units="s**-1")
        self.add_output("data:handling_qualities:lateral:modes:dutch_roll:damping_ratio")
        self.add_output("data:handling_qualities:lateral:modes:dutch_roll:undamped_frequency", units="s**-1")
        self.add_output("data:handling_qualities:lateral:modes:dutch_roll:period", units="s")
        self.add_output("data:handling_qualities:lateral:modes:roll:real_part", units="s**-1")
        self.add_output("data:handling_qualities:lateral:modes:roll:imag_part", units="s**-1")
        self.add_output("data:handling_qualities:lateral:modes:roll:time_half", units="s")
        self.add_output("data:handling_qualities:lateral:modes:roll:time_double", units="s")
        self.add_output("data:handling_qualities:lateral:modes:spiral:real_part", units="s**-1")
        self.add_output("data:handling_qualities:lateral:modes:spiral:imag_part", units="s**-1")
        self.add_output("data:handling_qualities:lateral:modes:spiral:time_half", units="s")
        self.add_output("data:handling_qualities:lateral:modes:spiral:time_double", units="s")



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
        u_s = inputs["data:reference_flight_condition:speed"]
        q_s = 0.5 * ro * u_s**2

        # mu = 2 * mass / (ro * S * b / 2)

        # TODO: moments and products of inertia computation
        Ix = inputs["data:weight:aircraft:inertia:Iox"]
        # Ix = Ix / (ro * S * (b / 2)**3)
        Iz = inputs["data:weight:aircraft:inertia:Ioz"]
        # Iz = Iz / (ro * S * (b / 2)**3)
        Jxz = inputs["data:weight:aircraft:inertia:Ioxz"]
        # Jxz = Jxz / (ro * S * (b / 2)**3)
        i_x = Jxz / Ix
        i_z = Jxz / Iz

        CY_beta = inputs["data:handling_qualities:lateral:derivatives:CY:beta"]
        CY_p = inputs["data:handling_qualities:lateral:derivatives:CY:rollrate"]
        CY_r = inputs["data:handling_qualities:lateral:derivatives:CY:yawrate"]

        Cl_beta = inputs["data:handling_qualities:lateral:derivatives:Cl:beta"]
        Cl_p = inputs["data:handling_qualities:lateral:derivatives:Cl:rollrate"]
        Cl_r = inputs["data:handling_qualities:lateral:derivatives:Cl:yawrate"]

        Cn_beta = inputs["data:handling_qualities:lateral:derivatives:Cn:beta"]
        Cn_p = inputs["data:handling_qualities:lateral:derivatives:Cn:rollrate"]
        Cn_r = inputs["data:handling_qualities:lateral:derivatives:Cn:yawrate"]

        Y_beta = q_s * S / u_s * CY_beta * u_s/mass
        L_beta = q_s * S / u_s * b * Cl_beta * u_s/Ix
        N_beta = q_s * S / u_s * b * Cn_beta * u_s/Iz
        Y_p = q_s * S * b / (2 * u_s) * CY_p / mass
        L_p = q_s * S / (2 * u_s) * b**2 * Cl_p / Ix
        N_p = q_s * S / (2 * u_s) * b**2 * Cn_p / Iz
        Y_r = q_s * S * b / (2 * u_s) * CY_r / mass
        L_r = q_s * S / (2 * u_s) * b**2 * Cl_r / Ix
        N_r = q_s * S / (2 * u_s) * b**2 * Cn_r / Iz

        g = 32.174   # gravity acceleration in imperial units (ft/s**2)

        a11 = float(Y_beta / u_s)
        a12 = float(Y_p / u_s)
        a13 = float(Y_r / u_s - 1.0)
        a14 = float(g * math.cos(theta_s) / u_s)

        a21 = float((L_beta + i_x * N_beta) / (1 - i_x * i_z))
        a22 = float((L_p + i_x * N_p) / (1 - i_x * i_z))
        a23 = float((L_r + i_x * N_r) / (1 - i_x * i_z))
        a24 = 0.0

        a31 = float((N_beta + i_z * L_beta) / (1 - i_x * i_z))
        a32 = float((N_p + i_z * L_p) / (1 - i_x * i_z))
        a33 = float((N_r + i_z * L_r) / (1 - i_x * i_z))
        a34 = 0.0

        a41 = 0.0
        a42 = 1.0
        a43 = 0.0
        a44 = 0.0

        A = np.array([[a11, a12, a13, a14],
                      [a21, a22, a23, a24],
                      [a31, a32, a33, a34],
                      [a41, a42, a43, a44]])


        ###### LATERAL-DIRECTIONAL MODES ######
        # TODO: raise exception or warning if any of the eigenvalues does not have the expected form (real or imaginary)
        # Obtaining eigenvalues from matrix A
        w, v = np.linalg.eig(A)
        w.sort()

        ## Dutch-roll mode ##
        w_dr = w[2]
        real_dr = w_dr.real
        imag_dr = w_dr.imag
        wn_dr = math.sqrt(real_dr**2 + imag_dr**2)
        damp_dr = - real_dr / math.sqrt(real_dr**2 + imag_dr**2)
        period_dr = 2*math.pi / imag_dr

        ## Roll mode ##
        w_roll = w[0]
        real_roll = w_roll.real
        imag_roll = w_roll.imag
        t_half_roll = - math.log(2) / real_roll
        t_double_roll = math.log(2) / real_roll

        ## Spiral mode ##
        w_spi = w[3]
        real_spi = w_spi.real
        imag_spi = w_spi.imag
        t_half_spi = - math.log(2) / real_spi
        t_double_spi = math.log(2) / real_spi

        outputs["data:handling_qualities:lateral:modes:dutch_roll:real_part"] = real_dr
        outputs["data:handling_qualities:lateral:modes:dutch_roll:imag_part"] = imag_dr
        outputs["data:handling_qualities:lateral:modes:dutch_roll:damping_ratio"] = damp_dr
        outputs["data:handling_qualities:lateral:modes:dutch_roll:undamped_frequency"] = wn_dr
        outputs["data:handling_qualities:lateral:modes:dutch_roll:period"] = period_dr
        outputs["data:handling_qualities:lateral:modes:roll:real_part"] = real_roll
        outputs["data:handling_qualities:lateral:modes:roll:imag_part"] = imag_roll
        outputs["data:handling_qualities:lateral:modes:roll:time_half"] = t_half_roll
        outputs["data:handling_qualities:lateral:modes:roll:time_double"] = t_double_roll
        outputs["data:handling_qualities:lateral:modes:spiral:real_part"] = real_spi
        outputs["data:handling_qualities:lateral:modes:spiral:imag_part"] = imag_spi
        outputs["data:handling_qualities:lateral:modes:spiral:time_half"] = t_half_spi
        outputs["data:handling_qualities:lateral:modes:spiral:time_double"] = t_double_spi


