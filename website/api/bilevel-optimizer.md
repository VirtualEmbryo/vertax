# Bilevel optimizer base class

`vertax.bilevelopt.bilevelopt`

`_BilevelOptimizer` holds the settings and methods that `PbcBilevelOptimizer` and
`BoundedBilevelOptimizer` share. Its name starts with an underscore because it is not meant to be
instantiated, but it is exported by `vertax` and its attributes are the ones you set on either
optimizer.

::: vertax.bilevelopt.bilevelopt

::: vertax.bilevelopt.bilevelopt._BilevelOptimizer
    options:
      show_root_heading: true
      show_root_full_path: false
