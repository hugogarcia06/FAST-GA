import logging
import math
import os.path as pth

import numpy as np
import openmdao.api as om
from pandas import read_csv

from scipy import interpolate

from fastga.models.handling_qualities import resources

_LOGGER = logging.getLogger(__name__)


@staticmethod
def model_figure_digitization_simple(variable_eje_x, variable_curvas):
    """
    Modelo para digitalizar datos de una graphica x-y con diferentes curvas para diferentes valores de un parametro.

    :param variable_eje_x:
    :param variable_curvas:
    """

    # TODO: modificar nombre fichero
    file = pth.join(resources.__path__[0], "nombre_fichero.csv")
    db = read_csv(file)

    x_curva_1 = db["CURVA_1_X"]
    y_curva_1 = db["CURVA_1_Y"]
    errors = np.logical_or(np.isnan(x_curva_1), np.isnan(y_curva_1))
    x_curva_1 = x_curva_1[np.logical_not(errors)].tolist()
    y_curva_1 = y_curva_1[np.logical_not(errors)].tolist()

    x_curva_2 = db["CURVA_2_X"]
    y_curva_2 = db["CURVA_2_Y"]
    errors = np.logical_or(np.isnan(x_curva_2), np.isnan(y_curva_2))
    x_curva_2 = x_curva_2[np.logical_not(errors)].tolist()
    y_curva_2 = y_curva_2[np.logical_not(errors)].tolist()

    x_curva_3 = db["CURVA_3_X"]
    y_curva_3 = db["CURVA_3_Y"]
    errors = np.logical_or(np.isnan(x_curva_3), np.isnan(y_curva_3))
    x_curva_3 = x_curva_3[np.logical_not(errors)].tolist()
    y_curva_3 = y_curva_3[np.logical_not(errors)].tolist()

    x_curva_4 = db["CURVA_4_X"]
    y_curva_4 = db["CURVA_4_Y"]
    errors = np.logical_or(np.isnan(x_curva_4), np.isnan(y_curva_4))
    x_curva_4 = x_curva_4[np.logical_not(errors)].tolist()
    y_curva_4 = y_curva_4[np.logical_not(errors)].tolist()

    x_curva_5 = db["CURVA_5_X"]
    y_curva_5 = db["CURVA_5_Y"]
    errors = np.logical_or(np.isnan(x_curva_5), np.isnan(y_curva_5))
    x_curva_5 = x_curva_5[np.logical_not(errors)].tolist()
    y_curva_5 = y_curva_5[np.logical_not(errors)].tolist()

    x_curva_6 = db["CURVA_6_X"]
    y_curva_6 = db["CURVA_6_Y"]
    errors = np.logical_or(np.isnan(x_curva_6), np.isnan(y_curva_6))
    x_curva_6 = x_curva_6[np.logical_not(errors)].tolist()
    y_curva_6 = y_curva_6[np.logical_not(errors)].tolist()

    x_curva_7 = db["CURVA_7_X"]
    y_curva_7 = db["CURVA_7_Y"]
    errors = np.logical_or(np.isnan(x_curva_7), np.isnan(y_curva_7))
    x_curva_7 = x_curva_7[np.logical_not(errors)].tolist()
    y_curva_7 = y_curva_7[np.logical_not(errors)].tolist()

    x_curva_8 = db["CURVA_8_X"]
    y_curva_8 = db["CURVA_8_Y"]
    errors = np.logical_or(np.isnan(x_curva_8), np.isnan(y_curva_8))
    x_curva_8 = x_curva_8[np.logical_not(errors)].tolist()
    y_curva_8 = y_curva_8[np.logical_not(errors)].tolist()

    x_curva_9 = db["CURVA_9_X"]
    y_curva_9 = db["CURVA_9_Y"]
    errors = np.logical_or(np.isnan(x_curva_9), np.isnan(y_curva_9))
    x_curva_9 = x_curva_9[np.logical_not(errors)].tolist()
    y_curva_9 = y_curva_9[np.logical_not(errors)].tolist()

    x_curva_10 = db["CURVA_10_X"]
    y_curva_10 = db["CURVA_10_Y"]
    errors = np.logical_or(np.isnan(x_curva_10), np.isnan(y_curva_10))
    x_curva_10 = x_curva_10[np.logical_not(errors)].tolist()
    y_curva_10 = y_curva_10[np.logical_not(errors)].tolist()

    k_curva_1 = interpolate.interp1d(x_curva_1, y_curva_1)
    k_curva_2 = interpolate.interp1d(x_curva_2, y_curva_2)
    k_curva_3 = interpolate.interp1d(x_curva_3, y_curva_3)
    k_curva_4 = interpolate.interp1d(x_curva_4, y_curva_4)
    k_curva_5 = interpolate.interp1d(x_curva_5, y_curva_5)
    k_curva_6 = interpolate.interp1d(x_curva_6, y_curva_6)
    k_curva_7 = interpolate.interp1d(x_curva_7, y_curva_7)
    k_curva_8 = interpolate.interp1d(x_curva_8, y_curva_8)
    k_curva_9 = interpolate.interp1d(x_curva_9, y_curva_9)
    k_curva_10 = interpolate.interp1d(x_curva_10, y_curva_10)

    if (
            (variable_eje_x != np.clip(variable_eje_x, min(x_curva_1), max(x_curva_1)))
            or (variable_eje_x != np.clip(variable_eje_x, min(x_curva_2), max(x_curva_2)))
            or (variable_eje_x != np.clip(variable_eje_x, min(x_curva_3), max(x_curva_3)))
            or (variable_eje_x != np.clip(variable_eje_x, min(x_curva_4), max(x_curva_4)))
            or (variable_eje_x != np.clip(variable_eje_x, min(x_curva_5), max(x_curva_5)))
            or (variable_eje_x != np.clip(variable_eje_x, min(x_curva_6), max(x_curva_6)))
            or (variable_eje_x != np.clip(variable_eje_x, min(x_curva_7), max(x_curva_7)))
            or (variable_eje_x != np.clip(variable_eje_x, min(x_curva_8), max(x_curva_8)))
            or (variable_eje_x != np.clip(variable_eje_x, min(x_curva_9), max(x_curva_9)))
            or (variable_eje_x != np.clip(variable_eje_x, min(x_curva_10), max(x_curva_10)))
    ):
        # TODO: modify warning
        _LOGGER.warning("Flap angle value outside of the range in Roskam's book, value clipped")

    k_curva = [
        float(k_curva_1(np.clip(variable_eje_x, min(x_curva_1), max(x_curva_1)))),
        float(k_curva_2(np.clip(variable_eje_x, min(x_curva_2), max(x_curva_2)))),
        float(k_curva_3(np.clip(variable_eje_x, min(x_curva_3), max(x_curva_3)))),
        float(k_curva_4(np.clip(variable_eje_x, min(x_curva_4), max(x_curva_4)))),
        float(k_curva_5(np.clip(variable_eje_x, min(x_curva_5), max(x_curva_5)))),
        float(k_curva_6(np.clip(variable_eje_x, min(x_curva_6), max(x_curva_6)))),
        float(k_curva_7(np.clip(variable_eje_x, min(x_curva_7), max(x_curva_7)))),
        float(k_curva_8(np.clip(variable_eje_x, min(x_curva_8), max(x_curva_8)))),
        float(k_curva_9(np.clip(variable_eje_x, min(x_curva_9), max(x_curva_9)))),
        float(k_curva_10(np.clip(variable_eje_x, min(x_curva_10), max(x_curva_10)))),

    ]

    # TODO: revisar el rango de valores de los parametros
    if variable_curvas != np.clip(variable_curvas, 0.1, 0.5):
        # TODO: modify warning
        _LOGGER.warning(
            "Chord ratio value outside of the range in Roskam's book, value clipped"
        )

    # TODO: revisar el rango de valores de los parametros
    valor_eje_y = float(
        interpolate.interp1d([0.1, 0.15, 0.25, 0.3, 0.4, 0.5], k_curva)(
            np.clip(variable_curvas, 0.1, 0.5)
        )
    )

    return valor_eje_y


@staticmethod
def model_figure_digitization_triple(variable_eje_x, variable_curvas, variable_graphicas):
    """
    Modelo para digitalizar datos de tres graphicas x-y con diferentes curvas para diferentes valores de un parametro.

    :param variable_eje_x:
    :param variable_curvas:
    :param variable_graphicas:
    """

    # ----- GRAFICA 1 -----
    # TODO: modificar nombre fichero donde estan los datos de la graphica 1
    file = pth.join(resources.__path__[0], "nombre_fichero_graphica_1.csv")
    db = read_csv(file)

    x_curva_1_graph_1 = db["CURVA_1_X"]
    y_curva_1_graph_1 = db["CURVA_1_Y"]
    errors = np.logical_or(np.isnan(x_curva_1_graph_1), np.isnan(y_curva_1_graph_1))
    x_curva_1_graph_1 = x_curva_1_graph_1[np.logical_not(errors)].tolist()
    y_curva_1_graph_1 = y_curva_1_graph_1[np.logical_not(errors)].tolist()

    x_curva_2_graph_1 = db["CURVA_2_X"]
    y_curva_2_graph_1 = db["CURVA_2_Y"]
    errors = np.logical_or(np.isnan(x_curva_2_graph_1), np.isnan(y_curva_2_graph_1))
    x_curva_2_graph_1 = x_curva_2_graph_1[np.logical_not(errors)].tolist()
    y_curva_2_graph_1 = y_curva_2_graph_1[np.logical_not(errors)].tolist()

    x_curva_3_graph_1 = db["CURVA_3_X"]
    y_curva_3_graph_1 = db["CURVA_3_Y"]
    errors = np.logical_or(np.isnan(x_curva_3_graph_1), np.isnan(y_curva_3_graph_1))
    x_curva_3_graph_1 = x_curva_3_graph_1[np.logical_not(errors)].tolist()
    y_curva_3_graph_1 = y_curva_3_graph_1[np.logical_not(errors)].tolist()

    x_curva_4_graph_1 = db["CURVA_4_X"]
    y_curva_4_graph_1 = db["CURVA_4_Y"]
    errors = np.logical_or(np.isnan(x_curva_4_graph_1), np.isnan(y_curva_4_graph_1))
    x_curva_4_graph_1 = x_curva_4_graph_1[np.logical_not(errors)].tolist()
    y_curva_4_graph_1 = y_curva_4_graph_1[np.logical_not(errors)].tolist()

    x_curva_5_graph_1 = db["CURVA_5_X"]
    y_curva_5_graph_1 = db["CURVA_5_Y"]
    errors = np.logical_or(np.isnan(x_curva_5_graph_1), np.isnan(y_curva_5_graph_1))
    x_curva_5_graph_1 = x_curva_5_graph_1[np.logical_not(errors)].tolist()
    y_curva_5_graph_1 = y_curva_5_graph_1[np.logical_not(errors)].tolist()

    x_curva_6_graph_1 = db["CURVA_6_X"]
    y_curva_6_graph_1 = db["CURVA_6_Y"]
    errors = np.logical_or(np.isnan(x_curva_6_graph_1), np.isnan(y_curva_6_graph_1))
    x_curva_6_graph_1 = x_curva_6_graph_1[np.logical_not(errors)].tolist()
    y_curva_6_graph_1 = y_curva_6_graph_1[np.logical_not(errors)].tolist()

    x_curva_7_graph_1 = db["CURVA_7_X"]
    y_curva_7_graph_1 = db["CURVA_7_Y"]
    errors = np.logical_or(np.isnan(x_curva_7_graph_1), np.isnan(y_curva_7_graph_1))
    x_curva_7_graph_1 = x_curva_7_graph_1[np.logical_not(errors)].tolist()
    y_curva_7_graph_1 = y_curva_7_graph_1[np.logical_not(errors)].tolist()

    x_curva_8_graph_1 = db["CURVA_8_X"]
    y_curva_8_graph_1 = db["CURVA_8_Y"]
    errors = np.logical_or(np.isnan(x_curva_8_graph_1), np.isnan(y_curva_8_graph_1))
    x_curva_8_graph_1 = x_curva_8_graph_1[np.logical_not(errors)].tolist()
    y_curva_8_graph_1 = y_curva_8_graph_1[np.logical_not(errors)].tolist()

    x_curva_9_graph_1 = db["CURVA_9_X"]
    y_curva_9_graph_1 = db["CURVA_9_Y"]
    errors = np.logical_or(np.isnan(x_curva_9_graph_1), np.isnan(y_curva_9_graph_1))
    x_curva_9_graph_1 = x_curva_9_graph_1[np.logical_not(errors)].tolist()
    y_curva_9_graph_1 = y_curva_9_graph_1[np.logical_not(errors)].tolist()

    x_curva_10_graph_1 = db["CURVA_10_X"]
    y_curva_10_graph_1 = db["CURVA_10_Y"]
    errors = np.logical_or(np.isnan(x_curva_10_graph_1), np.isnan(y_curva_10_graph_1))
    x_curva_10_graph_1 = x_curva_10_graph_1[np.logical_not(errors)].tolist()
    y_curva_10_graph_1 = y_curva_10_graph_1[np.logical_not(errors)].tolist()

    k_curva_1_graph_1 = interpolate.interp1d(x_curva_1_graph_1, y_curva_1_graph_1)
    k_curva_2_graph_1 = interpolate.interp1d(x_curva_2_graph_1, y_curva_2_graph_1)
    k_curva_3_graph_1 = interpolate.interp1d(x_curva_3_graph_1, y_curva_3_graph_1)
    k_curva_4_graph_1 = interpolate.interp1d(x_curva_4_graph_1, y_curva_4_graph_1)
    k_curva_5_graph_1 = interpolate.interp1d(x_curva_5_graph_1, y_curva_5_graph_1)
    k_curva_6_graph_1 = interpolate.interp1d(x_curva_6_graph_1, y_curva_6_graph_1)
    k_curva_7_graph_1 = interpolate.interp1d(x_curva_7_graph_1, y_curva_7_graph_1)
    k_curva_8_graph_1 = interpolate.interp1d(x_curva_8_graph_1, y_curva_8_graph_1)
    k_curva_9_graph_1 = interpolate.interp1d(x_curva_9_graph_1, y_curva_9_graph_1)
    k_curva_10_graph_1 = interpolate.interp1d(x_curva_10_graph_1, y_curva_10_graph_1)

    if (
            (variable_eje_x != np.clip(variable_eje_x, min(x_curva_1_graph_1), max(x_curva_1_graph_1)))
            or (variable_eje_x != np.clip(variable_eje_x, min(x_curva_2_graph_1), max(x_curva_2_graph_1)))
            or (variable_eje_x != np.clip(variable_eje_x, min(x_curva_3_graph_1), max(x_curva_3_graph_1)))
            or (variable_eje_x != np.clip(variable_eje_x, min(x_curva_4_graph_1), max(x_curva_4_graph_1)))
            or (variable_eje_x != np.clip(variable_eje_x, min(x_curva_5_graph_1), max(x_curva_5_graph_1)))
            or (variable_eje_x != np.clip(variable_eje_x, min(x_curva_6_graph_1), max(x_curva_6_graph_1)))
            or (variable_eje_x != np.clip(variable_eje_x, min(x_curva_7_graph_1), max(x_curva_7_graph_1)))
            or (variable_eje_x != np.clip(variable_eje_x, min(x_curva_8_graph_1), max(x_curva_8_graph_1)))
            or (variable_eje_x != np.clip(variable_eje_x, min(x_curva_9_graph_1), max(x_curva_9_graph_1)))
            or (variable_eje_x != np.clip(variable_eje_x, min(x_curva_10_graph_1), max(x_curva_10_graph_1)))
    ):
        # TODO: modify warning
        _LOGGER.warning("Flap angle value outside of the range in Roskam's book, value clipped")

    k_curva_graph_1 = [
        float(k_curva_1_graph_1(np.clip(variable_eje_x, min(x_curva_1_graph_1), max(x_curva_1_graph_1)))),
        float(k_curva_2_graph_1(np.clip(variable_eje_x, min(x_curva_2_graph_1), max(x_curva_2_graph_1)))),
        float(k_curva_3_graph_1(np.clip(variable_eje_x, min(x_curva_3_graph_1), max(x_curva_3_graph_1)))),
        float(k_curva_4_graph_1(np.clip(variable_eje_x, min(x_curva_4_graph_1), max(x_curva_4_graph_1)))),
        float(k_curva_5_graph_1(np.clip(variable_eje_x, min(x_curva_5_graph_1), max(x_curva_5_graph_1)))),
        float(k_curva_6_graph_1(np.clip(variable_eje_x, min(x_curva_6_graph_1), max(x_curva_6_graph_1)))),
        float(k_curva_7_graph_1(np.clip(variable_eje_x, min(x_curva_7_graph_1), max(x_curva_7_graph_1)))),
        float(k_curva_8_graph_1(np.clip(variable_eje_x, min(x_curva_8_graph_1), max(x_curva_8_graph_1)))),
        float(k_curva_9_graph_1(np.clip(variable_eje_x, min(x_curva_9_graph_1), max(x_curva_9_graph_1)))),
        float(k_curva_10_graph_1(np.clip(variable_eje_x, min(x_curva_10_graph_1), max(x_curva_10_graph_1)))),

    ]

    # TODO: revisar el rango de valores de los parametros
    if variable_curvas != np.clip(variable_curvas, 0.1, 0.5):
        # TODO: modify warning
        _LOGGER.warning(
            "Chord ratio value outside of the range in Roskam's book, value clipped"
        )

    # TODO: revisar el rango de valores de los parametros
    valor_eje_y_graph_1 = float(
        interpolate.interp1d([0.1, 0.15, 0.25, 0.3, 0.4, 0.5], k_curva_graph_1)(
            np.clip(variable_curvas, 0.1, 0.5)
        )
    )

    # ----- GRAFICA 2 -----
    # TODO: modificar nombre fichero donde estan los datos de la graphica 2
    file = pth.join(resources.__path__[0], "nombre_fichero_graphica_2.csv")
    db = read_csv(file)

    x_curva_1_graph_2 = db["CURVA_1_X"]
    y_curva_1_graph_2 = db["CURVA_1_Y"]
    errors = np.logical_or(np.isnan(x_curva_1_graph_2), np.isnan(y_curva_1_graph_2))
    x_curva_1_graph_2 = x_curva_1_graph_2[np.logical_not(errors)].tolist()
    y_curva_1_graph_2 = y_curva_1_graph_2[np.logical_not(errors)].tolist()

    x_curva_2_graph_2 = db["CURVA_2_X"]
    y_curva_2_graph_2 = db["CURVA_2_Y"]
    errors = np.logical_or(np.isnan(x_curva_2_graph_2), np.isnan(y_curva_2_graph_2))
    x_curva_2_graph_2 = x_curva_2_graph_2[np.logical_not(errors)].tolist()
    y_curva_2_graph_2 = y_curva_2_graph_2[np.logical_not(errors)].tolist()

    x_curva_3_graph_2 = db["CURVA_3_X"]
    y_curva_3_graph_2 = db["CURVA_3_Y"]
    errors = np.logical_or(np.isnan(x_curva_3_graph_2), np.isnan(y_curva_3_graph_2))
    x_curva_3_graph_2 = x_curva_3_graph_2[np.logical_not(errors)].tolist()
    y_curva_3_graph_2 = y_curva_3_graph_2[np.logical_not(errors)].tolist()

    x_curva_4_graph_2 = db["CURVA_4_X"]
    y_curva_4_graph_2 = db["CURVA_4_Y"]
    errors = np.logical_or(np.isnan(x_curva_4_graph_2), np.isnan(y_curva_4_graph_2))
    x_curva_4_graph_2 = x_curva_4_graph_2[np.logical_not(errors)].tolist()
    y_curva_4_graph_2 = y_curva_4_graph_2[np.logical_not(errors)].tolist()

    x_curva_5_graph_2 = db["CURVA_5_X"]
    y_curva_5_graph_2 = db["CURVA_5_Y"]
    errors = np.logical_or(np.isnan(x_curva_5_graph_2), np.isnan(y_curva_5_graph_2))
    x_curva_5_graph_2 = x_curva_5_graph_2[np.logical_not(errors)].tolist()
    y_curva_5_graph_2 = y_curva_5_graph_2[np.logical_not(errors)].tolist()

    x_curva_6_graph_2 = db["CURVA_6_X"]
    y_curva_6_graph_2 = db["CURVA_6_Y"]
    errors = np.logical_or(np.isnan(x_curva_6_graph_2), np.isnan(y_curva_6_graph_2))
    x_curva_6_graph_2 = x_curva_6_graph_2[np.logical_not(errors)].tolist()
    y_curva_6_graph_2 = y_curva_6_graph_2[np.logical_not(errors)].tolist()

    x_curva_7_graph_2 = db["CURVA_7_X"]
    y_curva_7_graph_2 = db["CURVA_7_Y"]
    errors = np.logical_or(np.isnan(x_curva_7_graph_2), np.isnan(y_curva_7_graph_2))
    x_curva_7_graph_2 = x_curva_7_graph_2[np.logical_not(errors)].tolist()
    y_curva_7_graph_2 = y_curva_7_graph_2[np.logical_not(errors)].tolist()

    x_curva_8_graph_2 = db["CURVA_8_X"]
    y_curva_8_graph_2 = db["CURVA_8_Y"]
    errors = np.logical_or(np.isnan(x_curva_8_graph_2), np.isnan(y_curva_8_graph_2))
    x_curva_8_graph_2 = x_curva_8_graph_2[np.logical_not(errors)].tolist()
    y_curva_8_graph_2 = y_curva_8_graph_2[np.logical_not(errors)].tolist()

    x_curva_9_graph_2 = db["CURVA_9_X"]
    y_curva_9_graph_2 = db["CURVA_9_Y"]
    errors = np.logical_or(np.isnan(x_curva_9_graph_2), np.isnan(y_curva_9_graph_2))
    x_curva_9_graph_2 = x_curva_9_graph_2[np.logical_not(errors)].tolist()
    y_curva_9_graph_2 = y_curva_9_graph_2[np.logical_not(errors)].tolist()

    x_curva_10_graph_2 = db["CURVA_10_X"]
    y_curva_10_graph_2 = db["CURVA_10_Y"]
    errors = np.logical_or(np.isnan(x_curva_10_graph_2), np.isnan(y_curva_10_graph_2))
    x_curva_10_graph_2 = x_curva_10_graph_2[np.logical_not(errors)].tolist()
    y_curva_10_graph_2 = y_curva_10_graph_2[np.logical_not(errors)].tolist()

    k_curva_1_graph_2 = interpolate.interp1d(x_curva_1_graph_2, y_curva_1_graph_2)
    k_curva_2_graph_2 = interpolate.interp1d(x_curva_2_graph_2, y_curva_2_graph_2)
    k_curva_3_graph_2 = interpolate.interp1d(x_curva_3_graph_2, y_curva_3_graph_2)
    k_curva_4_graph_2 = interpolate.interp1d(x_curva_4_graph_2, y_curva_4_graph_2)
    k_curva_5_graph_2 = interpolate.interp1d(x_curva_5_graph_2, y_curva_5_graph_2)
    k_curva_6_graph_2 = interpolate.interp1d(x_curva_6_graph_2, y_curva_6_graph_2)
    k_curva_7_graph_2 = interpolate.interp1d(x_curva_7_graph_2, y_curva_7_graph_2)
    k_curva_8_graph_2 = interpolate.interp1d(x_curva_8_graph_2, y_curva_8_graph_2)
    k_curva_9_graph_2 = interpolate.interp1d(x_curva_9_graph_2, y_curva_9_graph_2)
    k_curva_10_graph_2 = interpolate.interp1d(x_curva_10_graph_2, y_curva_10_graph_2)

    if (
            (variable_eje_x != np.clip(variable_eje_x, min(x_curva_1_graph_2), max(x_curva_1_graph_2)))
            or (variable_eje_x != np.clip(variable_eje_x, min(x_curva_2_graph_2), max(x_curva_2_graph_2)))
            or (variable_eje_x != np.clip(variable_eje_x, min(x_curva_3_graph_2), max(x_curva_3_graph_2)))
            or (variable_eje_x != np.clip(variable_eje_x, min(x_curva_4_graph_2), max(x_curva_4_graph_2)))
            or (variable_eje_x != np.clip(variable_eje_x, min(x_curva_5_graph_2), max(x_curva_5_graph_2)))
            or (variable_eje_x != np.clip(variable_eje_x, min(x_curva_6_graph_2), max(x_curva_6_graph_2)))
            or (variable_eje_x != np.clip(variable_eje_x, min(x_curva_7_graph_2), max(x_curva_7_graph_2)))
            or (variable_eje_x != np.clip(variable_eje_x, min(x_curva_8_graph_2), max(x_curva_8_graph_2)))
            or (variable_eje_x != np.clip(variable_eje_x, min(x_curva_9_graph_2), max(x_curva_9_graph_2)))
            or (variable_eje_x != np.clip(variable_eje_x, min(x_curva_10_graph_2), max(x_curva_10_graph_2)))
    ):
        # TODO: modify warning
        _LOGGER.warning("Flap angle value outside of the range in Roskam's book, value clipped")

    k_curva_graph_2 = [
        float(k_curva_1_graph_2(np.clip(variable_eje_x, min(x_curva_1_graph_2), max(x_curva_1_graph_2)))),
        float(k_curva_2_graph_2(np.clip(variable_eje_x, min(x_curva_2_graph_2), max(x_curva_2_graph_2)))),
        float(k_curva_3_graph_2(np.clip(variable_eje_x, min(x_curva_3_graph_2), max(x_curva_3_graph_2)))),
        float(k_curva_4_graph_2(np.clip(variable_eje_x, min(x_curva_4_graph_2), max(x_curva_4_graph_2)))),
        float(k_curva_5_graph_2(np.clip(variable_eje_x, min(x_curva_5_graph_2), max(x_curva_5_graph_2)))),
        float(k_curva_6_graph_2(np.clip(variable_eje_x, min(x_curva_6_graph_2), max(x_curva_6_graph_2)))),
        float(k_curva_7_graph_2(np.clip(variable_eje_x, min(x_curva_7_graph_2), max(x_curva_7_graph_2)))),
        float(k_curva_8_graph_2(np.clip(variable_eje_x, min(x_curva_8_graph_2), max(x_curva_8_graph_2)))),
        float(k_curva_9_graph_2(np.clip(variable_eje_x, min(x_curva_9_graph_2), max(x_curva_9_graph_2)))),
        float(k_curva_10_graph_2(np.clip(variable_eje_x, min(x_curva_10_graph_2), max(x_curva_10_graph_2)))),

    ]

    # TODO: revisar el rango de valores de los parametros
    if variable_curvas != np.clip(variable_curvas, 0.1, 0.5):
        # TODO: modify warning
        _LOGGER.warning(
            "Chord ratio value outside of the range in Roskam's book, value clipped"
        )

    # TODO: revisar el rango de valores de los parametros
    valor_eje_y_graph_2 = float(
        interpolate.interp1d([0.1, 0.15, 0.25, 0.3, 0.4, 0.5], k_curva_graph_2)(
            np.clip(variable_curvas, 0.1, 0.5)
        )
    )

    # ----- GRAFICA 3 -----
    # TODO: modificar nombre fichero donde estan los datos de la graphica 3
    file = pth.join(resources.__path__[0], "nombre_fichero_graphica_3.csv")
    db = read_csv(file)

    x_curva_1_graph_3 = db["CURVA_1_X"]
    y_curva_1_graph_3 = db["CURVA_1_Y"]
    errors = np.logical_or(np.isnan(x_curva_1_graph_3), np.isnan(y_curva_1_graph_3))
    x_curva_1_graph_3 = x_curva_1_graph_3[np.logical_not(errors)].tolist()
    y_curva_1_graph_3 = y_curva_1_graph_3[np.logical_not(errors)].tolist()

    x_curva_2_graph_3 = db["CURVA_2_X"]
    y_curva_2_graph_3 = db["CURVA_2_Y"]
    errors = np.logical_or(np.isnan(x_curva_2_graph_3), np.isnan(y_curva_2_graph_3))
    x_curva_2_graph_3 = x_curva_2_graph_3[np.logical_not(errors)].tolist()
    y_curva_2_graph_3 = y_curva_2_graph_3[np.logical_not(errors)].tolist()

    x_curva_3_graph_3 = db["CURVA_3_X"]
    y_curva_3_graph_3 = db["CURVA_3_Y"]
    errors = np.logical_or(np.isnan(x_curva_3_graph_3), np.isnan(y_curva_3_graph_3))
    x_curva_3_graph_3 = x_curva_3_graph_3[np.logical_not(errors)].tolist()
    y_curva_3_graph_3 = y_curva_3_graph_3[np.logical_not(errors)].tolist()

    x_curva_4_graph_3 = db["CURVA_4_X"]
    y_curva_4_graph_3 = db["CURVA_4_Y"]
    errors = np.logical_or(np.isnan(x_curva_4_graph_3), np.isnan(y_curva_4_graph_3))
    x_curva_4_graph_3 = x_curva_4_graph_3[np.logical_not(errors)].tolist()
    y_curva_4_graph_3 = y_curva_4_graph_3[np.logical_not(errors)].tolist()

    x_curva_5_graph_3 = db["CURVA_5_X"]
    y_curva_5_graph_3 = db["CURVA_5_Y"]
    errors = np.logical_or(np.isnan(x_curva_5_graph_3), np.isnan(y_curva_5_graph_3))
    x_curva_5_graph_3 = x_curva_5_graph_3[np.logical_not(errors)].tolist()
    y_curva_5_graph_3 = y_curva_5_graph_3[np.logical_not(errors)].tolist()

    x_curva_6_graph_3 = db["CURVA_6_X"]
    y_curva_6_graph_3 = db["CURVA_6_Y"]
    errors = np.logical_or(np.isnan(x_curva_6_graph_3), np.isnan(y_curva_6_graph_3))
    x_curva_6_graph_3 = x_curva_6_graph_3[np.logical_not(errors)].tolist()
    y_curva_6_graph_3 = y_curva_6_graph_3[np.logical_not(errors)].tolist()

    x_curva_7_graph_3 = db["CURVA_7_X"]
    y_curva_7_graph_3 = db["CURVA_7_Y"]
    errors = np.logical_or(np.isnan(x_curva_7_graph_3), np.isnan(y_curva_7_graph_3))
    x_curva_7_graph_3 = x_curva_7_graph_3[np.logical_not(errors)].tolist()
    y_curva_7_graph_3 = y_curva_7_graph_3[np.logical_not(errors)].tolist()

    x_curva_8_graph_3 = db["CURVA_8_X"]
    y_curva_8_graph_3 = db["CURVA_8_Y"]
    errors = np.logical_or(np.isnan(x_curva_8_graph_3), np.isnan(y_curva_8_graph_3))
    x_curva_8_graph_3 = x_curva_8_graph_3[np.logical_not(errors)].tolist()
    y_curva_8_graph_3 = y_curva_8_graph_3[np.logical_not(errors)].tolist()

    x_curva_9_graph_3 = db["CURVA_9_X"]
    y_curva_9_graph_3 = db["CURVA_9_Y"]
    errors = np.logical_or(np.isnan(x_curva_9_graph_3), np.isnan(y_curva_9_graph_3))
    x_curva_9_graph_3 = x_curva_9_graph_3[np.logical_not(errors)].tolist()
    y_curva_9_graph_3 = y_curva_9_graph_3[np.logical_not(errors)].tolist()

    x_curva_10_graph_3 = db["CURVA_10_X"]
    y_curva_10_graph_3 = db["CURVA_10_Y"]
    errors = np.logical_or(np.isnan(x_curva_10_graph_3), np.isnan(y_curva_10_graph_3))
    x_curva_10_graph_3 = x_curva_10_graph_3[np.logical_not(errors)].tolist()
    y_curva_10_graph_3 = y_curva_10_graph_3[np.logical_not(errors)].tolist()

    k_curva_1_graph_3 = interpolate.interp1d(x_curva_1_graph_3, y_curva_1_graph_3)
    k_curva_2_graph_3 = interpolate.interp1d(x_curva_2_graph_3, y_curva_2_graph_3)
    k_curva_3_graph_3 = interpolate.interp1d(x_curva_3_graph_3, y_curva_3_graph_3)
    k_curva_4_graph_3 = interpolate.interp1d(x_curva_4_graph_3, y_curva_4_graph_3)
    k_curva_5_graph_3 = interpolate.interp1d(x_curva_5_graph_3, y_curva_5_graph_3)
    k_curva_6_graph_3 = interpolate.interp1d(x_curva_6_graph_3, y_curva_6_graph_3)
    k_curva_7_graph_3 = interpolate.interp1d(x_curva_7_graph_3, y_curva_7_graph_3)
    k_curva_8_graph_3 = interpolate.interp1d(x_curva_8_graph_3, y_curva_8_graph_3)
    k_curva_9_graph_3 = interpolate.interp1d(x_curva_9_graph_3, y_curva_9_graph_3)
    k_curva_10_graph_3 = interpolate.interp1d(x_curva_10_graph_3, y_curva_10_graph_3)

    if (
            (variable_eje_x != np.clip(variable_eje_x, min(x_curva_1_graph_3), max(x_curva_1_graph_3)))
            or (variable_eje_x != np.clip(variable_eje_x, min(x_curva_2_graph_3), max(x_curva_2_graph_3)))
            or (variable_eje_x != np.clip(variable_eje_x, min(x_curva_3_graph_3), max(x_curva_3_graph_3)))
            or (variable_eje_x != np.clip(variable_eje_x, min(x_curva_4_graph_3), max(x_curva_4_graph_3)))
            or (variable_eje_x != np.clip(variable_eje_x, min(x_curva_5_graph_3), max(x_curva_5_graph_3)))
            or (variable_eje_x != np.clip(variable_eje_x, min(x_curva_6_graph_3), max(x_curva_6_graph_3)))
            or (variable_eje_x != np.clip(variable_eje_x, min(x_curva_7_graph_3), max(x_curva_7_graph_3)))
            or (variable_eje_x != np.clip(variable_eje_x, min(x_curva_8_graph_3), max(x_curva_8_graph_3)))
            or (variable_eje_x != np.clip(variable_eje_x, min(x_curva_9_graph_3), max(x_curva_9_graph_3)))
            or (variable_eje_x != np.clip(variable_eje_x, min(x_curva_10_graph_3), max(x_curva_10_graph_3)))
    ):
        # TODO: modify warning
        _LOGGER.warning("Flap angle value outside of the range in Roskam's book, value clipped")

    k_curva_graph_3 = [
        float(k_curva_1_graph_3(np.clip(variable_eje_x, min(x_curva_1_graph_3), max(x_curva_1_graph_3)))),
        float(k_curva_2_graph_3(np.clip(variable_eje_x, min(x_curva_2_graph_3), max(x_curva_2_graph_3)))),
        float(k_curva_3_graph_3(np.clip(variable_eje_x, min(x_curva_3_graph_3), max(x_curva_3_graph_3)))),
        float(k_curva_4_graph_3(np.clip(variable_eje_x, min(x_curva_4_graph_3), max(x_curva_4_graph_3)))),
        float(k_curva_5_graph_3(np.clip(variable_eje_x, min(x_curva_5_graph_3), max(x_curva_5_graph_3)))),
        float(k_curva_6_graph_3(np.clip(variable_eje_x, min(x_curva_6_graph_3), max(x_curva_6_graph_3)))),
        float(k_curva_7_graph_3(np.clip(variable_eje_x, min(x_curva_7_graph_3), max(x_curva_7_graph_3)))),
        float(k_curva_8_graph_3(np.clip(variable_eje_x, min(x_curva_8_graph_3), max(x_curva_8_graph_3)))),
        float(k_curva_9_graph_3(np.clip(variable_eje_x, min(x_curva_9_graph_3), max(x_curva_9_graph_3)))),
        float(k_curva_10_graph_3(np.clip(variable_eje_x, min(x_curva_10_graph_3), max(x_curva_10_graph_3)))),

    ]

    # TODO: revisar el rango de valores de los parametros
    if variable_curvas != np.clip(variable_curvas, 0.1, 0.5):
        # TODO: modify warning
        _LOGGER.warning(
            "Chord ratio value outside of the range in Roskam's book, value clipped"
        )

    # TODO: revisar el rango de valores de los parametros
    valor_eje_y_graph_3 = float(
        interpolate.interp1d([0.1, 0.15, 0.25, 0.3, 0.4, 0.5], k_curva_graph_3)(
            np.clip(variable_curvas, 0.1, 0.5)
        )
    )

    # INTERPOLACION ENTRE LAS 3 GRÁFICAS
    valores_graphicas = [
        float(valor_eje_y_graph_1),
        float(valor_eje_y_graph_2),
        float(valor_eje_y_graph_3),
    ]
    # TODO: revisar rango de valores
    valor_interpolado_graphicas = float(
        interpolate.interp1d([0.1, 0.15, 0.5], valores_graphicas)(
            np.clip(variable_graphicas, 0.1, 0.5)
        )
    )

    return valor_interpolado_graphicas
