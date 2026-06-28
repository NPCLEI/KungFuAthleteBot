from holosoma_min.config_types.logger import DisabledLoggerConfig, WandbLoggerConfig

disabled = DisabledLoggerConfig()

wandb = WandbLoggerConfig(mode="online")

wandb_offline = WandbLoggerConfig(mode="offline")

DEFAULTS = {
    "disabled": disabled,
    "wandb": wandb,
    "wandb_offline": wandb_offline,
}
