# Contribute

For the docstrings we use the [numpy style](https://numpydoc.readthedocs.io/en/latest/format.html).

You can install some of the commonly used development tools by using e3nn's 'dev' extra:

```
pip install -e '.[dev]'
```

To have atomic code style checks performed at each commit, you can install the pre-commit hook using:

```
pre-commit install
```

These checks are automatically run on any commit made to the github repository but the pre-commit hook allows you to see if there are any problems locally.

Additionally, you may want to run the tests locally before pushing to remote.  This can be done with (from the root e3nn directory):

```
pytest tests
```

For formatting we use the [black](https://black.readthedocs.io/en/stable/index.html) library.
It can be installed with:

```
pip install black
```

and run with:

```
black .
```
