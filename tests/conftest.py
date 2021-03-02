# For good practice, we *should* do this:
# See https://docs.pytest.org/en/stable/fixture.html#using-fixtures-from-other-projects
# pytest_plugins = ['e3nn.util.test']
# But doing so exposes float_tolerance to doctests, which don't support parametrized, autouse fixtures.
# Importing directly somehow only brings in the fixture later, preventing the issue.
from e3nn.util.test import float_tolerance
# Suppress linter errors
float_tolerance = float_tolerance
