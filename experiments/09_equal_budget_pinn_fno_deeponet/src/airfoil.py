"""The airfoil problem of this experiment: the implementation lives in the PINNeAPPle library
(``pinneapple_analysis.benchmark.builtin.airfoil_impl``: data split, MLP ensemble, MeshGraphNet, POD + MLP fields,
potential-flow and RANS PINNs, scores). This module points it at this experiment's prepared airfoil data
(``results/prepared/airfoil``, filled by ``bench09.migrate`` from ``data/airfoil`` and ``data/airfoil_ood``)."""
from common import RES

from pinneapple_analysis.benchmark.builtin import airfoil_impl as _impl
from pinneapple_analysis.benchmark.builtin.airfoil_impl import *  # noqa: F401,F403

_impl.set_data_dir(RES / "prepared" / "airfoil")
