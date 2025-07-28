import os

input_directory = "/mnt/c/c++_tests/games_work/ml_work/Raw_data/S22by7_additional/"   
output_directory = "/mnt/c/c++_tests/games_work/ML_work/clean_data/S22by7/"

def extract_Sij_Rij():
   for file_name in os.listdir(input_directory):
      if os.path.isfile(os.path.join(input_directory, file_name)):
         with open(input_directory + file_name, 'r') as in_file:
            output_file = output_directory + file_name[0:file_name.find("-unCP")] + ".csv"
            with open(output_file, 'w') as out_file:
               out_file.write("Sij,Rij\n")
               counter = 0
               for line in in_file:
                  line = line.strip()
                  if line.startswith("|SIJ| and RIJ"):
                     fields = line.split()
                     out_file.write(fields[5] + ",")
                     out_file.write(fields[6] + "\n")
                     counter +=1
               if counter == 0: 
                  print("File " + file_name + " has no SIJ/RIJ values.")
                  os.remove(output_file)
                  print(f"File '{output_file}' deleted successfully.")                                   

if __name__ == "__main__":
   extract_Sij_Rij()