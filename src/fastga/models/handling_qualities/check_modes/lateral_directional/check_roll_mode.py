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

import os
import openmdao.api as om
import numpy as np
import matplotlib.pyplot as plt
from importlib.resources import path
from ... import resources as local_resources

from openmdao.utils.file_wrap import InputFileGenerator


class CheckRollMode(om.ExplicitComponent):
    """
    # TODOC:
    """
    def initialize(self):
        self.options.declare("result_folder_path", default="", types=str)

    def setup(self):
        self.add_input("data:geometry:aircraft:class", val=np.nan)
        self.add_input("data:reference_flight_condition:flight_phase_category", val=np.nan)

        self.add_input("data:handling_qualities:lateral:modes:roll:real_part", val=np.nan, units="s**-1")
        self.add_input("data:handling_qualities:lateral:modes:roll:imag_part", val=np.nan,
                       units="s**-1")

        self.add_output(
            "data:handling_qualities:lateral:modes:roll:check:time_constant:satisfaction_level")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        results_folder_path = self.options["result_folder_path"]

        aircraft_class = inputs["data:geometry:aircraft:class"]
        flight_phase_category = inputs["data:reference_flight_condition:flight_phase_category"]

        real_roll = inputs["data:handling_qualities:lateral:modes:roll:real_part"]
        imag_roll = inputs["data:handling_qualities:lateral:modes:roll:imag_part"]

        time_constant_roll = - 1 / real_roll

        # GET REQUIREMENTS
        tc_roll_req = self.get_roll_mode_requirements(aircraft_class, flight_phase_category)

        # CHECK REQUIREMENTS
        check_time_constant = self.check_roll_mode_requirements(time_constant_roll, tc_roll_req)

        # PLOT S-PLANE
        self.plot_s_plane(real_roll, imag_roll, tc_roll_req, results_folder_path)

        # WRITE RESULTS
        self.write_results(real_roll, imag_roll, time_constant_roll, check_time_constant, results_folder_path)

        outputs["data:handling_qualities:lateral:modes:roll:check:time_constant:satisfaction_level"] = check_time_constant


    @staticmethod
    def get_roll_mode_requirements(aircraft_class, flight_phase_category):
        """
        The roll mode time constant Tr shall be no greater than the appropriate value from the regulations.

        """

        tc_roll_req_1 = 0.0
        tc_roll_req_2 = 0.0
        tc_roll_req_3 = 0.0
        if flight_phase_category == 1.0:
            if aircraft_class == 1.0 or aircraft_class == 4.0:
                tc_roll_req_1 = 1.0
                tc_roll_req_2 = 1.4
                tc_roll_req_3 = 10.0
            elif aircraft_class == 2.0 or aircraft_class == 3.0:
                tc_roll_req_1 = 1.4
                tc_roll_req_2 = 3.0
                tc_roll_req_3 = 10.0
        elif flight_phase_category == 2.0:
            tc_roll_req_1 = 1.4
            tc_roll_req_2 = 3.0
            tc_roll_req_3 = 10.0
        elif flight_phase_category == 3.0:
            if aircraft_class == 1.0 or aircraft_class == 4.0:
                tc_roll_req_1 = 1.0
                tc_roll_req_2 = 1.4
                tc_roll_req_3 = 10.0
            elif aircraft_class == 2.0 or aircraft_class == 3.0:
                tc_roll_req_1 = 1.4
                tc_roll_req_2 = 3.0
                tc_roll_req_3 = 10.0

        tc_roll_req = [tc_roll_req_1, tc_roll_req_2, tc_roll_req_3]

        return tc_roll_req

    @staticmethod
    def check_roll_mode_requirements(tc_roll, tc_roll_req):

        tc_roll_req_1 = tc_roll_req[0]
        tc_roll_req_2 = tc_roll_req[1]
        tc_roll_req_3 = tc_roll_req[2]

        ### CHECK ###
        # Time constant
        check_time_constant = 0.0
        if tc_roll < tc_roll_req_1:
            check_time_constant = 1.0
        elif tc_roll_req_1 <= tc_roll < tc_roll_req_2:
            check_time_constant = 2.0
        elif (tc_roll_req_3 != 0.0) and (tc_roll_req_2 <= tc_roll <= tc_roll_req_3):
            check_time_constant = 3.0

        return check_time_constant


    @staticmethod
    def plot_s_plane(real_roll, imag_roll, tc_roll_req, results_folder_path):

        def get_vertical_limits(time_constant):

            real_roll = -1.0 / time_constant
            x = real_roll * np.ones(1000)
            y = np.linspace(-100, 100, 1000)

            return x, y

        tc_roll_req_1 = tc_roll_req[0]
        tc_roll_req_2 = tc_roll_req[1]
        tc_roll_req_3 = tc_roll_req[2]

        # fig, ax = plt.subplots(figsize=(11.2, 8.4))
        fig, ax = plt.subplots()
        ax.set_title("Check of Roll Mode Characteristics Versus Flying Quality Requirements")
        ax.set_xlabel(r"$n$")
        ax.set_ylabel(r"$jw$")
        ax.grid(True, which="both")
        # x-axis
        ax.plot(np.linspace(-1000, 1000, 1000), 0.0 * np.linspace(-1000, 1000, 1000), color="black")
        # y-axis
        ax.plot(0.0 * np.linspace(-1000, 1000, 1000), np.linspace(-1000, 1000, 1000), color="black")
        ax.scatter(real_roll, imag_roll, label=r"$\lambda_{roll}$")

        # Damping-frequency product limit
        x_1, y_1 = get_vertical_limits(tc_roll_req_1)
        ax.plot(x_1, y_1, linestyle="--", color="red", label="Level 1")
        x_2, y_2 = get_vertical_limits(tc_roll_req_2)
        ax.plot(x_2, y_2, linestyle="--", color="orange", label="Level 2")
        if tc_roll_req_3 != 0.0:
            x_3, y_3 = get_vertical_limits(tc_roll_req_3)
            ax.plot(x_3, y_3, linestyle="--", color="yellow", label="Level 3")

        ax.legend(loc="upper right")
        ax.set_xbound(real_roll * 3, 0.1)
        ax.set_ybound(-real_roll * 3, real_roll * 3)
        ax.set_xbound(-11.0, 2.0)
        ax.set_ybound(-5.0, 5.0)
        # plt.show()

        results_dir = results_folder_path
        plots_dir = os.path.join(results_dir, 'Check Modes Plots/')
        plot_name = "roll_mode_s-plane.png"

        if not os.path.isdir(plots_dir):
            os.makedirs(plots_dir)

        fig.savefig(plots_dir + plot_name)


    @staticmethod
    def write_results(real_part, imag_part, time_constant, check_time_constant, results_folder_path):

        file_name = "check_modes_results_roll.txt"
        resources_directory = "C:/Users/hugog/OneDrive/Escritorio/FAST-GA/src/fastga/models/handling_qualities/resources"
        saving_directory = os.path.join(resources_directory, file_name)

        parser = InputFileGenerator()
        with path(local_resources, "check_modes_results_dr.txt") as input_template_path:
            parser.set_template_file(str(input_template_path))
            parser.set_generated_file(saving_directory)
            parser.mark_anchor("roll_real_part")
            parser.transfer_var(round(float(real_part), 5), 0, 3)
            parser.mark_anchor("roll_imag_part")
            parser.transfer_var(round(float(imag_part), 5), 0, 3)
            parser.mark_anchor("roll_time_constant")
            parser.transfer_var(round(float(time_constant), 5), 0, 3)
            parser.mark_anchor("roll_time_constant_level")
            parser.transfer_var(round(float(check_time_constant), 5), 0, 3)
            parser.generate()

        os.remove(
            os.path.join(resources_directory, "check_modes_results_dr.txt")
        )


