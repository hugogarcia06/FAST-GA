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
import matplotlib.pyplot as plt
import os


class PlotAircraftModes(om.ExplicitComponent):
    def initialize(self):
        self.options.declare("result_folder_path", default="", types=str)

    def setup(self):
        self.add_input("data:handling_qualities:longitudinal:modes:phugoid:real_part", val=np.nan, units="s**-1")
        self.add_input("data:handling_qualities:longitudinal:modes:phugoid:imag_part", val=np.nan, units="s**-1")
        self.add_input("data:handling_qualities:longitudinal:modes:short_period:real_part", val=np.nan, units="s**-1")
        self.add_input("data:handling_qualities:longitudinal:modes:short_period:imag_part", val=np.nan, units="s**-1")
        self.add_input("data:handling_qualities:lateral:modes:dutch_roll:real_part", val=np.nan, units="s**-1")
        self.add_input("data:handling_qualities:lateral:modes:dutch_roll:imag_part", val=np.nan, units="s**-1")
        self.add_input("data:handling_qualities:lateral:modes:roll:real_part", val=np.nan, units="s**-1")
        self.add_input("data:handling_qualities:lateral:modes:roll:imag_part", val=np.nan, units="s**-1")
        self.add_input("data:handling_qualities:lateral:modes:spiral:real_part", units="s**-1")
        self.add_input("data:handling_qualities:lateral:modes:spiral:imag_part", units="s**-1")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        real_ph = inputs["data:handling_qualities:longitudinal:modes:phugoid:real_part"]
        imag_ph = inputs["data:handling_qualities:longitudinal:modes:phugoid:imag_part"]
        real_sp = inputs["data:handling_qualities:longitudinal:modes:short_period:real_part"]
        imag_sp = inputs["data:handling_qualities:longitudinal:modes:short_period:imag_part"]

        real_dr = inputs["data:handling_qualities:lateral:modes:dutch_roll:real_part"]
        imag_dr = inputs["data:handling_qualities:lateral:modes:dutch_roll:imag_part"]
        real_roll = inputs["data:handling_qualities:lateral:modes:roll:real_part"]
        imag_roll = inputs["data:handling_qualities:lateral:modes:roll:imag_part"]
        real_spiral = inputs["data:handling_qualities:lateral:modes:spiral:real_part"]
        imag_spiral = inputs["data:handling_qualities:lateral:modes:spiral:imag_part"]

        real_parts = [real_ph, real_sp, real_dr, real_roll, real_spiral]
        imag_parts = [imag_ph, imag_sp, imag_dr, imag_roll, imag_spiral]

        ### PLOT ###
        # fig, ax = plt.subplots(figsize=(11.2, 8.4))
        fig, ax = plt.subplots()
        ax.set_title("Aircraft Modes in the s-plane")
        ax.set_xlabel(r"$n$")
        ax.set_ylabel(r"$jw$")
        ax.grid(visible=True, which="both")
        # x-axis
        ax.plot(np.linspace(-10, 10, 1000), 0.0*np.linspace(-10, 10, 1000), color="black")
        # y-axis
        ax.plot(0.0*np.linspace(-10, 10, 1000), np.linspace(-10, 10, 1000), color="black")

        ax.scatter(real_ph, imag_ph, marker="$p$", color="blue", label=r"$\lambda_{ph}$")
        ax.scatter(real_ph, -imag_ph, marker="$p$", color="blue", label="")
        ax.scatter(real_sp, imag_sp, marker="$s$", color="blue", label=r"$\lambda_{sp}$")
        ax.scatter(real_sp, -imag_sp, marker="$s$", color="blue", label="")
        ax.scatter(real_dr, imag_dr, marker="$d$", color="orange", label=r"$\lambda_{dr}$")
        ax.scatter(real_dr, -imag_dr, marker="$d$", color="orange", label="")
        ax.scatter(real_roll, imag_roll, marker="$r$", color="orange", label=r"$\lambda_{roll}$")
        ax.scatter(real_spiral, imag_spiral, marker="$s$", color="orange", label=r"$\lambda_{spiral}$")

        ax.legend(loc="upper right")
        min_real_parts = min(real_parts)
        max_real_parts = max(real_parts)
        max_imag_parts = max(imag_parts)
        # ax.set_xbound(min_real_parts*1.25, max(abs(max_real_parts), 1.0)*1.5)
        # ax.set_ybound(-max_imag_parts*1.5, max_imag_parts*1.5)
        ax.set_xbound(-11.0, 2.5)
        ax.set_ybound(-7.0, 7.0)
       # plt.show()

        results_dir = self.options["result_folder_path"]
        plots_dir = os.path.join(results_dir, 'Check Modes Plots/')
        plot_name = "aircraftmodes_s-plane.png"

        if not os.path.isdir(plots_dir):
            os.makedirs(plots_dir)

        fig.savefig(plots_dir + plot_name)

