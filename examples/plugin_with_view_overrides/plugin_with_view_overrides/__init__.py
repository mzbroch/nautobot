try:
    from importlib import metadata
except ImportError:
    # Python version < 3.8
    import importlib_metadata as metadata

__version__ = metadata.version(__name__)

from nautobot.extras.plugins import PluginConfig


class PluginWithOverridesConfig(PluginConfig):
    name = "plugin_with_view_overrides"
    verbose_name = "Plugin With View Overrides"
    author = "Nautobot development team"
    author_email = "nautobot@example.com"
    version = __version__
    description = "For testing purposes only"
    base_url = "plugin-with-view-overrides"


config = PluginWithOverridesConfig

default_app_config = "nautobot.core.apps.CoreConfig"
