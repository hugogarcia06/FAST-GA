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
import matplotlib.pyplot as plt


class CheckPhugoid(om.ExplicitComponent):
    """
    # TODOC:
    """
    def setup(self):
        self.add_input("data:handling_qualities:longitudinal:modes:phugoid:real_part", val=np.nan, units="s**-1")
        self.add_input("data:handling_qualities:longitudinal:modes:phugoid:imag_part", val=np.nan, units="s**-1")

        self.add_output("data:handling_qualities:longitudinal:modes:phugoid:check:damping_ratio:satisfaction_level")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        real_ph = inputs["data:handling_qualities:longitudinal:modes:phugoid:real_part"]
        imag_ph = inputs["data:handling_qualities:longitudinal:modes:phugoid:imag_part"]

        wn_ph = math.sqrt(real_ph**2 + imag_ph**2)
        z_ph = - real_ph / math.sqrt(real_ph**2 + imag_ph**2)

        ### PHUGOID REQUIREMENTS ###
        # Level 1
        z_ph_req_1 = 0.04

        # Level 2
        z_ph_req_2 = 0.0

        # Level 3
        t_2_ph_3 = 55
        n_req_3 = math.log(2) / t_2_ph_3

        ### CHECK ###
        check_phugoid_damping = 0.0
        if z_ph >= z_ph_req_1:
            check_phugoid_damping = 1.0
        elif z_ph_req_2 <= z_ph < z_ph_req_1:
            check_phugoid_damping = 2.0
        elif real_ph <= n_req_3:
            check_phugoid_damping = 3.0

        ### PLOT ###
        self.plot_s_plane(real_ph, imag_ph, z_ph_req_1, z_ph_req_2, n_req_3)

        outputs["data:handling_qualities:longitudinal:modes:phugoid:check:damping_ratio:satisfaction_level"] = check_phugoid_damping


    @staticmethod
    def plot_s_plane(real_ph, imag_ph, z_ph_req_1, z_ph_req_2, n_req_3):

        def get_damping_limits(damping_ratio):
            phi = math.asin(damping_ratio)
            phi = math.pi / 2 - phi
            x = np.linspace(-100, 100, 100)
            y = math.tan(-phi) * x

            return x, y

        def get_vertical_limits(z_wn_product):
            x = -z_wn_product * np.ones(1000)
            y = np.linspace(-100, 100, 1000)

            return x, y

        ### PLOT ###
        fig, ax = plt.subplots()
        ax.set_title("Check of Phugoid Characteristics Versus Flying Quality Requirements")
        ax.set_xlabel(r"$n$")
        ax.set_ylabel(r"$jw$")
        ax.grid(visible=True, which="both")
        # x-axis
        ax.plot(np.linspace(-10, 10, 1000), 0.0*np.linspace(-10, 10, 1000), color="black")
        # y-axis
        ax.plot(0.0*np.linspace(-10, 10, 1000), np.linspace(-10, 10, 1000), color="black")

        ax.scatter(real_ph, imag_ph, label=r"$\lambda_{ph}$")

        # Damping ratio limits
        x_1, y_1 = get_damping_limits(z_ph_req_1)
        ax.plot(x_1, y_1, linestyle="--", color="red", label="Level 1")
        x_2, y_2 = get_damping_limits(z_ph_req_2)
        ax.plot(x_2, y_2, linestyle="--", color="orange", label="Level 2")
        x_3, y_3 = get_vertical_limits(-n_req_3)
        ax.plot(x_3, y_3, linestyle="--", color="yellow", label="Level 3")

        ax.legend(loc="upper right")
        ax.set_xbound(real_ph*4, n_req_3 * 4/3)
        ax.set_ybound(-imag_ph, imag_ph*4)
        plt.show()
