#S22 Dataset Feature Extraction Information

The current idea is to characterize each S22 output file by:

* Its undamped couloumb energy. This is found in the first column for each file.
The way this is obtained from the raw .log files is:

$Undamped Coulomb/Electrostatic Energy = CHARGE-CHARGE + CHARGE-DIPOLE + CHARGE-QUADRUPOLE + CHARGE-OCTUPOLE + DIPOLE-DIPOLE + DIPOLE-QUADRUPOLE + QUADRUPOLE-QUADRUPOLE$

For reference:
$Total Coulomb/Electrostatic Energy = Undamped Coul. E. + Damping function term (aka OVERLAP PENETRATION ENERGY)$

* S_ij and R_ij values. These values range from 4x4 (16 Sij and Rij values) to 21x21 (441 Sij and Rij values). 
Files with fewer Sij/Rij features were padded with zeroes to match the dimensions of the largest file.

## General Characteristics

For the S22 dataset, each file has 882 Sij/Rij values, plus an undamped coulomb energy term.
Then, we represent each file as an 883-dimensional vector.

## Hartree to Kcal/mol conversion
The conversion factor being used is 1 hartree = 627.5094740631 kcal/mol.
Source: https://en.wikipedia.org/wiki/Hartree


Data obtained on: 2025-06-26 03:06:45