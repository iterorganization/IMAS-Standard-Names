# Guidelines for Construction of IMAS Standard Names

## Characters

Standard names consist of lower-lettters, digits and underscores, and begin with
a letter. Upper case is not used.

## Spelling

US spelling is used, e.g. analyze, center.

## Qualifications

Standard names may be qualified by the addition of phrases in certain standard
forms and order. These qualifications do not change the units of the quantity. 

[[component](#component)] standard_name [at [position](#position)] [due to
[process](#process)]

### Component

The direction of the spatial component of a vector is indicated by one of the
words `radial`, `vertical`, `toroidal` or `poloidal`, in the cylindrical or
toroidal/poloidal coordinate system.

The direction of a vector relative to the local magnetic field may be indicated
by the words `parallel` or `diamagnetic`.

### Position

A phrase `at_<position>` may be used to indicate the value of a quantity at a
predefined position. Some examples are `at_boundary`, `at_magnetic_axis` and
`at_current_center`.

### Process

The specification of a physical process by the phrase `due_to_<process>` means
that the quantity named is a single term in a sum of terms which together
compose the general quantity named by omitting the phrase. Some examples are
`due_to_conduction`, `due_to_convection`.

## Transformations

Standard names may be derived from other standard names (represented here by X,
Y and Z) by the following rules. Successive transformations may be applied.
Transformations may alter the units as shown.

{{ read_csv('transformations.csv') }}

## Generic names

The following names are used with consistent meanings and units as elements in
other standard names, although they are themselves too general to be chosen as
standard names. They are recorded here for reference only. *These are not
standard names*.

{{ read_csv('generic_names.csv') }}

[^1]:
    Temperature of plasma species (e.g. `electron_temperature`) is expressed in
    `eV`, other temperatures (e.g. `wall_temparature`) are given in `K`.
