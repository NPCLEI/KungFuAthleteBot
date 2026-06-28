"""Default randomization manager configurations."""

from holosoma_min.config_values.loco.g1.randomization import g1_29dof_randomization
from holosoma_min.config_values.loco.t1.randomization import t1_29dof_randomization
from holosoma_min.config_values.wbt.g1.randomization import g1_29dof_wbt_randomization, g1_29dof_wbt_randomization_w_object

none = None

DEFAULTS = {
    "none": none,
    "t1_29dof": t1_29dof_randomization,
    "g1_29dof": g1_29dof_randomization,
    "g1_29dof_wbt": g1_29dof_wbt_randomization,
    "g1_29dof_wbt_w_object": g1_29dof_wbt_randomization_w_object,
}
