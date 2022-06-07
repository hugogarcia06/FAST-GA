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

import logging
import math
import os.path as pth

import numpy as np
import openmdao.api as om
from pandas import read_csv

from scipy import interpolate

from fastga.models.weight import resources

_LOGGER = logging.getLogger(__name__)


class FigureDigitization(om.ExplicitComponent):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.phase = None

    @staticmethod
    def get_k1(y_cg, b, cr, ct) -> float:
        """
        Data from USAF DATCOM to estimate wing rolling correlation K factor presented in Figure 8.1-23.

        :param y_cg: lateral centroidal distance of half-wing from aircraft plane of symmetry
        :param b: wing span
        :param cr: wing root chord
        :param ct: wing tip chord

        """

        y_ratio = 6*y_cg / (b * (cr + 2*ct)/(cr + ct))

        file = pth.join(resources.__path__[0], "8_1-23.csv")
        db = read_csv(file)

        x = db["X"]
        y = db["Y"]
        errors = np.logical_or(np.isnan(x), np.isnan(y))
        x = x[np.logical_not(errors)].tolist()
        y = y[np.logical_not(errors)].tolist()

        if float(y_ratio) != np.clip(float(y_ratio), min(x), max(x)):
            _LOGGER.warning("X-axis value outside of the range in DATCOM, value clipped")

        k1 = float(interpolate.interp1d(x, y)(np.clip(float(y_ratio), min(x), max(x))))

        return k1


    @staticmethod
    def get_k2(x_cg, l) -> float:
        """
        Data from USAF DATCOM to estimate the fuselage pitching correlation K factor presented in Figure 8.1-24.

        :param x_cg: longitudinal centroidal distance of the fuselage from the nose.
        :param l: fuselage length

        """

        x_ratio = abs(l/2.0 - x_cg) / (l/2.0)

        file = pth.join(resources.__path__[0], "8_1-24.csv")
        db = read_csv(file)

        x = db["X"]
        y = db["Y"]
        errors = np.logical_or(np.isnan(x), np.isnan(y))
        x = x[np.logical_not(errors)].tolist()
        y = y[np.logical_not(errors)].tolist()

        if float(x_ratio) != np.clip(float(x_ratio), min(x), max(x)):
            _LOGGER.warning("X-axis value outside of the range in DATCOM, value clipped")

        k2 = float(interpolate.interp1d(x, y)(np.clip(float(x_ratio), min(x), max(x))))

        return k2


    @staticmethod
    def get_k3(d, fus_weight, fus_struct_weight) -> float:
        """
        Data from USAF DATCOM to estimate the fuselage rolling correlation K factor presented in Figure 8.1-25.

        :param d: fuselage average diameter
        :param fus_weight: fuselage weight
        :param fus_struct_weight: fuselage structural weight (considering bulkheads, frames, stringers and skin).

        """

        ratio = d ** 0.5 * fus_struct_weight / fus_weight

        file = pth.join(resources.__path__[0], "8_1-25.csv")
        db = read_csv(file)

        x = db["X"]
        y = db["Y"]
        errors = np.logical_or(np.isnan(x), np.isnan(y))
        x = x[np.logical_not(errors)].tolist()
        y = y[np.logical_not(errors)].tolist()

        if float(ratio) != np.clip(float(ratio), min(x), max(x)):
            _LOGGER.warning("X-axis value outside of the range in DATCOM, value clipped")

        k3 = float(interpolate.interp1d(x, y)(np.clip(float(ratio), min(x), max(x))))

        return k3


    @staticmethod
    def get_k4(y_cg, b, cr, ct) -> float:
        """
        Data from USAF DATCOM to estimate the horizontal tail rolling correlation K factor presented in Figure 8.1-26.

        :param y_cg: lateral centroidal distance of half-tail from aircraft plane of symmetry
        :param b: horizontal tail span
        :param cr: horizontal tail root chord
        :param ct: horizontal tail tip chord

        """

        y_ratio = 6*y_cg / (b * (cr + 2*ct)/(cr + ct))

        file = pth.join(resources.__path__[0], "8_1-26.csv")
        db = read_csv(file)

        x = db["X"]
        y = db["Y"]
        errors = np.logical_or(np.isnan(x), np.isnan(y))
        x = x[np.logical_not(errors)].tolist()
        y = y[np.logical_not(errors)].tolist()

        if float(y_ratio) != np.clip(float(y_ratio), min(x), max(x)):
            _LOGGER.warning("X-axis value outside of the range in DATCOM, value clipped")

        k4 = float(interpolate.interp1d(x, y)(np.clip(float(y_ratio), min(x), max(x))))

        return k4


    @staticmethod
    def get_k5(z_cg, b, cr, ct) -> float:
        """
        Data from USAF DATCOM to estimate the vertical tail rolling correlation K factor presented in Figure 8.1-27.

        :param z_cg: vertical centroidal distance of vertica1 tail from the theoretical root chord (at fuselage)
        :param b: vertical tail span.
        :param cr: vertical tail root chord
        :param ct: vertical tail tip chord

        """

        z_ratio = 3*z_cg / (b * (cr + 2*ct)/(cr + ct))

        file = pth.join(resources.__path__[0], "8_1-27.csv")
        db = read_csv(file)

        x = db["X"]
        y = db["Y"]
        errors = np.logical_or(np.isnan(x), np.isnan(y))
        x = x[np.logical_not(errors)].tolist()
        y = y[np.logical_not(errors)].tolist()

        if float(z_ratio) != np.clip(float(z_ratio), min(x), max(x)):
            _LOGGER.warning("X-axis value outside of the range in DATCOM, value clipped")

        k5 = float(interpolate.interp1d(x, y)(np.clip(float(z_ratio), min(x), max(x))))

        return k5


