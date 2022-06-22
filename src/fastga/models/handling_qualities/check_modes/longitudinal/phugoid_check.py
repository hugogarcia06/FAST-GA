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
        self.add_input("data:handling_qualities:longitudinal:modes:phugoid:damping_ratio", val=np.nan)
        self.add_input("data:handling_qualities:longitudinal:modes:phugoid:undamped_frequency", val=np.nan, units="rad/s")

        self.add_output("data:handling_qualities:longitudinal:modes:phugoid:check:damping_ratio:satisfaction_level")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        z_ph = inputs["data:handling_qualities:longitudinal:modes:phugoid:damping_ratio"]
        wn_ph = inputs["data:handling_qualities:longitudinal:modes:phugoid:undamped_frequency"]

        real_ph = - z_ph * wn_ph
        imag_ph = wn_ph * math.sqrt(1 - z_ph**2)

        ### PHUGOID REQUIREMENTS ###
        # Level 1
        z_ph_req_1 = 0.04
        phi_req_1 = math.asin(z_ph_req_1)
        phi_req_1 = math.pi/2 - phi_req_1

        x1 = np.linspace(-1, 1, 100)
        y1 = math.tan(-phi_req_1) * x1

        # Level 2
        z_ph_req_2 = 0.0
        y2 = np.linspace(-1, 1, 100)
        x2 = 0.0 * y2

        # Level 3
        t_2_ph_3 = 55
        n_req_3 = math.log(2) / t_2_ph_3
        y3 = np.linspace(-1, 1, 100)
        x3 = np.ones(100) * n_req_3

        ### CHECK ###
        check_phugoid_damping = 0.0
        if z_ph <= z_ph_req_1:
            check_phugoid_damping = 1.0
        elif z_ph_req_1 <= z_ph <= z_ph_req_2:
            check_phugoid_damping = 2.0
        elif z_ph_req_2 <= real_ph <= n_req_3:
            check_phugoid_damping = 3.0

        ### PLOT ###
        fig, ax = plt.subplots()
        ax.set_title("Check of Phugoid Characteristics Versus Flying Quality Requirements")
        ax.set_xlabel(r"$n$")
        ax.set_ylabel(r"$jw$")
        ax.grid(visible=True, which="both")
        ax.scatter(real_ph, imag_ph, label=r"$\lambda_{ph}$")
        # x-axis
        ax.plot(np.linspace(-10, 10, 1000), 0.0*np.linspace(-10, 10, 1000), color="black")
        # y-axis
        ax.plot(0.0*np.linspace(-10, 10, 1000), np.linspace(-10, 10, 1000), color="black")

        ax.plot(x1, y1, linestyle="--", color="red", label="Level 1")
        ax.plot(x2, y2, linestyle="--", color="orange", label="Level 2")
        ax.plot(x3, y3, linestyle="--", color="yellow", label="Level 3")
        ax.legend(loc="upper right")
        ax.set_xbound(real_ph*4, n_req_3 * 4/3)
        ax.set_ybound(-imag_ph, imag_ph*4)
        plt.show()

        # TODO: what output? plot in a file? bool that say if requirements are satisfied?
        outputs["data:handling_qualities:longitudinal:modes:phugoid:check:damping_ratio:satisfaction_level"] = check_phugoid_damping
