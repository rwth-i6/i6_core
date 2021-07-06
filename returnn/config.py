__all__ = ["CodeWrapper", "ReturnnConfig", "WriteReturnnConfigJob"]

import base64
import inspect
import json
import pickle
import pprint
import string
import textwrap

from sisyphus import *
from sisyphus.delayed_ops import DelayedBase
from sisyphus.hash import sis_hash_helper

Path = setup_path(__package__)
Variable = tk.Variable


def instanciate_delayed(o):
    """
    Recursively traverses a structure and calls .get() on all
    existing Delayed Operations, especially Variables in the structure

    :param Any o: nested structure that may contain DelayedBase objects
    :return:
    """
    if isinstance(o, DelayedBase):
        o = o.get()
    elif isinstance(o, list):
        for k in range(len(o)):
            o[k] = instanciate_delayed(o[k])
    elif isinstance(o, tuple):
        o = tuple(instanciate_delayed(e) for e in o)
    elif isinstance(o, dict):
        for k in o:
            o[k] = instanciate_delayed(o[k])
    return o


class CodeWrapper:
    def __init__(self, code):
        self.code = code

    def __repr__(self):
        return self.code


class ReturnnConfig:
    """
    An object that manages a RETURNN config.

    It can be used to serialize python functions and class definitions directly from
    Sisyphus code and paste them into the RETURNN config file.

    """

    PYTHON_CODE = textwrap.dedent(
        """\
        #!rnn.py
    
        ${PROLOG}
    
        ${REGULAR_CONFIG}
    
        locals().update(**config)
    
        ${EPILOG}
        """
    )

    def __init__(
        self,
        config,
        post_config=None,
        *,
        python_prolog=None,
        python_prolog_hash=None,
        python_epilog="",
        python_epilog_hash=None,
    ):
        """

        :param dict config: dictionary of the RETURNN config variables that are hashed
        :param dict post_config: dictionary of the RETURNN config variables that are not hashed
        :param None|str|Callable|Class|tuple|list|dict python_prolog: str or structure containing str/callables/classes
            that should be pasted as code at the beginning of the config file
        :param str|None python_prolog_hash: sets a specific hash for the python_prolog
        :param None|str|Callable|Class|tuple|list|dict python_epilog: str or structure containing
            str/callables/classes that should be pasted as code at the end of the config file
        :param str|None python_epilog_hash: sets a specific hash for the python_epilog
        """
        self.config = config
        self.post_config = post_config if post_config is not None else {}
        self.python_prolog = python_prolog
        self.python_prolog_hash = (
            python_prolog_hash if python_prolog_hash is not None else python_prolog
        )
        self.python_epilog = python_epilog
        self.python_epilog_hash = (
            python_epilog_hash if python_epilog_hash is not None else python_epilog
        )

    def get(self, key, default=None):
        if key in self.post_config:
            return self.post_config[key]
        return self.config.get(key, default)

    def write(self, path):
        with open(path, "wt", encoding="utf-8") as f:
            f.write(self.serialize())

    def serialize(self):
        self.check_consistency()
        config = self.config
        config.update(self.post_config)

        config = instanciate_delayed(config)

        config_lines = []
        unreadable_data = {}

        pp = pprint.PrettyPrinter(indent=2, width=150)
        for k, v in sorted(config.items()):
            if pprint.isreadable(v):
                config_lines.append("%s = %s" % (k, pp.pformat(v)))
            else:
                unreadable_data[k] = v

        if len(unreadable_data) > 0:
            config_lines.append("import json")
            json_data = json.dumps(unreadable_data).replace('"', '\\"')
            config_lines.append('config = json.loads("%s")' % json_data)
        else:
            config_lines.append("config = {}")

        python_prolog_code = self.__parse_python(self.python_prolog)
        python_epilog_code = self.__parse_python(self.python_epilog)

        python_code = string.Template(self.PYTHON_CODE).substitute(
            {
                "PROLOG": python_prolog_code,
                "REGULAR_CONFIG": "\n".join(config_lines),
                "EPILOG": python_epilog_code,
            }
        )
        return python_code

    def __parse_python(self, code, name=None):
        if code is None:
            return ""
        if isinstance(code, str):
            return code
        if isinstance(code, (tuple, list)):
            return "\n".join(self.__parse_python(c) for c in code)
        if isinstance(code, dict):
            return "\n".join(self.__parse_python(v, name=k) for k, v in code.items())
        if inspect.isfunction(code):
            try:
                return inspect.getsource(code)
            except OSError:
                # cannot get source, e.g. code is a lambda
                assert name is not None
                args = [
                    code.__code__.co_argcount,
                    code.__code__.co_kwonlyargcount,
                    code.__code__.co_nlocals,
                    code.__code__.co_stacksize,
                    code.__code__.co_flags,
                    code.__code__.co_code,
                    code.__code__.co_consts,
                    code.__code__.co_names,
                    code.__code__.co_varnames,
                    code.__code__.co_filename,
                    code.__code__.co_name,
                    code.__code__.co_firstlineno,
                    code.__code__.co_lnotab,
                    code.__code__.co_freevars,
                    code.__code__.co_cellvars,
                ]
                compiled = base64.b64encode(pickle.dumps(args)).decode("utf8")
                return (
                    "import types; import base64; import pickle; "
                    'code = types.CodeType(*pickle.loads(base64.b64decode("%s".encode("utf8")))); '
                    '%s = types.FunctionType(code, globals(), "%s")'
                    % (compiled, name, code.__name__)
                )
        if inspect.isclass(code):
            return inspect.getsource(code)
        raise RuntimeError("Could not serialize %s" % code)

    def check_consistency(self):
        """
        check that there is no config key overwritten by post_config
        """
        for key in self.config:
            assert key not in self.post_config, (
                "%s in post_config would overwrite existing entry in config" % key
            )

    def _sis_hash(self):
        h = {
            "returnn_config": self.config,
            "python_epilog_hash": self.python_epilog_hash,
            "python_prolog_hash": self.python_prolog_hash,
        }
        return sis_hash_helper(h)


class WriteReturnnConfigJob(Job):
    """
    Writes a ReturnnConfig into a .config file
    """

    def __init__(self, returnn_config):
        """

        :param ReturnnConfig returnn_config:
        """
        assert isinstance(returnn_config, ReturnnConfig)

        self.returnn_config = returnn_config

        self.out_returnn_config_file = self.output_path("returnn.config")

    def tasks(self):
        yield Task("run", resume="run", mini_task=True)

    def run(self):
        self.returnn_config.write(self.out_returnn_config_file.get_path())
