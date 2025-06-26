import os
from datetime import datetime

directory = "/mnt/c/C++_TESTS/simcodes_/compMetalsAlex/compmetalsalex/files june 25"   
output_file = directory + "/data/extractedhartree.csv"

def extract_fields_hartree():
   with open(output_file, 'w') as out_file:
      out_file.write("File,Undamped Coulomb")
      for i in range(882):
         out_file.write(",")
      out_file.write("\n")
      for file_name in os.listdir(directory):
         if os.path.isfile(os.path.join(directory, file_name)):
            with open(file_name, 'r') as in_file:
               if(file_name[0:5] != "extra"):
                  out_file.write(file_name[0:file_name.find("-Disp8")] + ",")
                  undamped_coul = 0
                  for line in in_file:
                     line = line.strip()
                     if line.startswith("CHARGE-CHARGE"):
                        undamped_coul += float(line[25:])
                     elif line.startswith("CHARGE-DIPOLE"):
                        undamped_coul += float(line[25:])
                     elif line.startswith("CHARGE-QUADRUPOLE"):
                        undamped_coul += float(line[25:])
                     elif line.startswith("CHARGE-OCTUPOLE"):
                        undamped_coul += float(line[25:])
                     elif line.startswith("DIPOLE-DIPOLE"):
                        undamped_coul += float(line[25:])
                     elif line.startswith("DIPOLE-QUADRUPOLE"):
                        undamped_coul += float(line[25:])
                     elif line.startswith("QUADRUPOLE-QUADRUPOLE"):
                        undamped_coul += float(line[25:])
                        out_file.write(str(undamped_coul))

                  with open(file_name, 'r') as in_file:
                     counter = 0
                     for line in in_file:
                        line = line.strip()
                        if line.startswith("|SIJ| and RIJ"):
                           out_file.write("," + line[25:41])
                           out_file.write("," + line[45:61])
                           counter += 1
                  while counter < 441:
                     out_file.write(",0,0")
                     counter += 1
                  out_file.write("\n")

output_file2 = directory + "/data/extractedkcal.csv"

def extract_fields_kcal():
   with open(output_file2, 'w') as out_file:
      out_file.write("File,Undamped Coulomb")
      for i in range(882):
         out_file.write(",")
      out_file.write("\n")
      for file_name in os.listdir(directory):
         if os.path.isfile(os.path.join(directory, file_name)):
            with open(file_name, 'r') as in_file:
               if(file_name[0:5] != "extra"):
                  out_file.write(file_name[0:file_name.find("-Disp8")] + ",")
                  
                  undamped_coul = 0
                  for line in in_file:
                     line = line.strip()
                     if line.startswith("CHARGE-CHARGE"):
                        undamped_coul += float(line[25:])*627.5094740631
                     elif line.startswith("CHARGE-DIPOLE"):
                        undamped_coul += float(line[25:])*627.5094740631
                     elif line.startswith("CHARGE-QUADRUPOLE"):
                        undamped_coul += float(line[25:])*627.5094740631
                     elif line.startswith("CHARGE-OCTUPOLE"):
                        undamped_coul += float(line[25:])*627.5094740631
                     elif line.startswith("DIPOLE-DIPOLE"):
                        undamped_coul += float(line[25:])*627.5094740631
                     elif line.startswith("DIPOLE-QUADRUPOLE"):
                        undamped_coul += float(line[25:])*627.5094740631
                     elif line.startswith("QUADRUPOLE-QUADRUPOLE"):
                        undamped_coul += float(line[25:])*627.5094740631
                        out_file.write(str(undamped_coul))

                  with open(file_name, 'r') as in_file:
                     counter = 0
                     for line in in_file:
                        line = line.strip()
                        if line.startswith("|SIJ| and RIJ"):
                           out_file.write("," + str(float(line[25:41])*627.5094740631))
                           out_file.write("," + str(float(line[45:61])*627.5094740631))
                           counter += 1
                  while counter < 441:
                     out_file.write(",0,0")
                     counter += 1
                  out_file.write("\n")


# def add_readme():
#    output_file3 = directory + "/data/readme.txt"
#    with open(output_file3, 'w') as out_file:
#       out_file.write('''The conversion factor being used is 1 hartree = 627.5094740631 kcal/mol.
# Source: https://en.wikipedia.org/wiki/Hartree\n
# Each csv file contains the undamped couloumb/electrostatic energy, and the S_ij and R_ij \
# values used for the damping function calculation for each sample.\n
# These values range from 4x4 (16 Sij and Rij values) to 21x21 (441 Sij and Rij values).\n
# Undamped Coulomb/Electrostatic Energy = CHARGE-CHARGE + CHARGE-DIPOLE + CHARGE-QUADRUPOLE + CHARGE-OCTUPOLE + DIPOLE-DIPOLE + DIPOLE-QUADRUPOLE + QUADRUPOLE-QUADRUPOLE\n\n
# Total Coulomb/Electrostatic Energy = Undamped Coul. E. + Damping function term (aka OVERLAP PENETRATION ENERGY)\n\n
# Data obtained on: ''' + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))                                    

if __name__ == "__main__":
   extract_fields_kcal()
   extract_fields_hartree()
   #add_readme()