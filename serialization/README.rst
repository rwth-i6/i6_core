Serialization is commonly used/needed for RETURNN configs.

TL;DR: If in doubt and writing new pipeline code, choose the last option (:class:`i6_core.returnn.ReturnnConfigV2`).

We have various options now:

- :mod:`i6_experiments.common.setups.serialization` /
  :mod:`i6_core.serialization` (copy of :mod:`i6_experiments.common.setups.serialization`):

  There is a :class:`SerializerObject` base class, deriving from Sisyphus :class:`DelayedBase`.
  There is also a :class:`Collection` to collect multiple such objects.
  The most commonly used serializer object is maybe the :class:`Import`,
  which imports some function from some module,
  also making sure that the path of the module is added to ``sys.path``,
  and properly setting a hash based on the module name,
  potentially also using the ``unhashed_package_root`` logic.

  For every serializer object, you can choose whether it is part of the hash or not.
  And if it is part of the hash, how the hash is defined exactly
  (``unhashed_package_root``, ``ignore_import_as_for_hash``, etc.).

  Those objects are intended to be put as the ``python_prolog`` or ``python_epilog``
  in a :class:`ReturnnConfig`.

  As an example, see :func:`i6_experiments.users.zeyer.train_v3.train`,
  :func:`i6_experiments.users.zeyer.recog.search_dataset`,
  :func:`i6_experiments.users.zeyer.forward_to_hdf.forward_to_hdf`.

- :class:`returnn_common.nn.ReturnnConfigSerializer`

  Used to serialize dim tags (:class:`Dim`)
  (and more, also the RETURNN-common ``nn.Module`` instances,
  transforming those into a RETURNN TF net dict, and handling dim tag refs properly;
  but we only use it for the dim tag serialization now).

  Specifically, we mostly just use :func:`ReturnnConfigSerializer.get_base_extern_data_py_code_str_direct`
  to generate the code for ``extern_data``.
  This function :func:`get_base_extern_data_py_code_str_direct`
  uses :class:`returnn_common.nn.ReturnnDimTagsProxy` internally.

  As an example, see :func:`i6_experiments.users.zeyer.train_v3.train`,
  :func:`i6_experiments.users.zeyer.recog.search_dataset`,
  :func:`i6_experiments.users.zeyer.forward_to_hdf.forward_to_hdf`.
  (In all those example cases, the generated Python code for ``extern_data`` is wrapped in :class:`NonhashedCode`,
  thus it does not get any hash.
  The assumption is that the dataset is already part of the hash,
  so any variations of ``extern_data`` should not matter.
  If ``extern_data`` is wrong, it would just crash anyway.)

- :func:`i6_experiments.common.setups.returnn.serialization.get_serializable_config`

  Operates on an existing :class:`ReturnnConfig` instance,
  going through all the config entries, checking whether they can be serialized directly,
  and if not, moving them to the ``python_epilog``.

  This handles dim tags (:class:`Dim`) directly using :class:`returnn_common.nn.ReturnnDimTagsProxy`.
  The hash is defined by the generated Python code,
  thus we cannot change the Python code generation now in a way that it would change the Python code.

  It also handles functions (:class:`FunctionType`) by copying the function source code
  (just as :class:`ReturnnConfig` also does).
  Functions are wrapped via :class:`CodeFromFunction`,
  and hashing can be controlled via ``hash_full_python_code``.

  As an example, see :func:`i6_experiments.users.zeyer.train_v3.train`,
  :func:`i6_experiments.users.zeyer.recog.search_dataset`,
  :func:`i6_experiments.users.zeyer.forward_to_hdf.forward_to_hdf`.

- :class:`ReturnnConfig` itself, e.g. ``python_epilog``:

  There is no special logic for the ``config`` or ``post_config``.
  It basically uses ``repr``.
  So that will not directly work for any special objects (dim tags, functions, etc).

  However, for ``python_epilog`` (also ``python_prolog``),
  it accepts :class:`DelayedBase`, and thus any custom logic can be done this way
  (see :class:`SerializerObject` or :func:`get_serializable_config` above).
  Additionally, when it finds a function (:class:`FunctionType`) or class,
  it will copy the function/class source code.

  Regarding hashing, ``config`` is used as-is, by default (the way we normally use it),
  also ``python_epilog`` is used as-is.
  Most of the :class:`SerializerObject` define a custom Sisyphus hash.
  When a function/class is directly used in ``python_epilog`` (not via :class:`SerializerObject`),
  it uses the hash of the function/class directly.
  The hash of a function/class is defined via ``(obj.__module__, obj.__qualname__)``.

- :class:`i6_experiments.common.utils.dump_py_code.PythonCodeDumper`

  This serializes any Python object in the same way ``pickle`` does,
  but instead of generating pickled raw data,
  it creates equivalent Python code.
  This is very generic and should always work exactly when ``pickle`` works.
  The generated code looks very artificial, though,
  using ``obj = object.__new__(cls)`` and ``obj.__setstate__(...)``.

  This serialization is currently only used by
  :func:`i6_experiments.common.helpers.dependency_boundary.dependency_boundary`.

- :func:`i6_core.serialization.serialization_v2`/:class:`i6_core.returnn.ReturnnConfigV2`

  Very generic code which should handle the whole RETURNN config,
  i.e. dim tags, functions, classes, etc.,
  so that no other serialization code is needed in addition.
  This is in contrast with most of the other approaches,
  which need to be used in combination.
