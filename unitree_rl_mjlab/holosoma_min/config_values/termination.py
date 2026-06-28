"""Default termination manager configurations."""

from holosoma_min.config_values.loco.g1.termination import g1_29dof_termination
from holosoma_min.config_values.loco.t1.termination import t1_29dof_termination
from holosoma_min.config_values.wbt.g1.termination import g1_29dof_wbt_termination

none = None

DEFAULTS = {
    "none": none,
    "t1_29dof": t1_29dof_termination,
    "g1_29dof": g1_29dof_termination,
    "g1_29dof_wbt": g1_29dof_wbt_termination,
}
