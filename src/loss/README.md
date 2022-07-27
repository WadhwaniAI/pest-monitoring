# Pest Monitoring Loss Functions

This directory contains scripts related to loss function definition
and composition.

## Data Types ("dtypes")

Defines core data types used within the loss framework, and exposed in
return types to clients.

## Abstractions

Abstract classes from which all loss functions should
extend. Extending from these, instead of from `torch.nn.Module`
directly, keeps development and access consistent.

## Calculators

Classes that handle loss value calculation. In most cases individual
classes in this file encapsulate a well defined loss calculator;
Torch's implementation of cross entropy loss loss, for example. In
doing so they have a well defined singular purpose.

## Systems

Classes in this file handle composition of multiple loss calculators.

### Note (30 Sept 2021)

Classes in this file (`systems.py`) should be built using
`abstractions.py::CompositeLoss`. Right now they do not because
detection losses require box transformation to compute. That coupling
requires non-standard class design. If that requirement goes away,
system losses should use `CompositeLoss` directly, as outlined in the
examples file.

## Aggregations

Classes that handle loss aggregation. A reference is typically
maintained by a calculator. Implementations depend on the intention of
the author of the calculator.

## Transforms

Classes that handle box transformation. They exist as separate classes
to facilitate decoupling from loss. These classes in particular, and
this functionality in general, should probably live somewhere else
("src/data").

## Examples

A file that outlines the gist of how to use classes in this
module:

* Instantiation of calculators
* Composition of calculators into a system
* Loss calculation using the system
* Ways of using the loss return types

For more examples, see `/test/loss`
