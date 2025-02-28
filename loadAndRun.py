import model
import numpy as np
import sys
import subprocess

np.set_printoptions(precision=2, suppress=True)

def process_values(values, model_name):
    predictions = model.predict_with_model([values], model_name)
    print (round(predictions[0, 0],4), round(predictions[0, 1],4), round(predictions[0, 3],4), round(predictions[0, 4],4), len(values), round(values[0],4), round(values[1],4), round(values[2],4), round(values[3],4), round(values[4],4), round(values[5],4), round(values[6],4), round(values[7],4), round(values[8],4))
    return round(predictions[0, 0],4), round(predictions[0, 1],4), round(predictions[0, 3],4), round(predictions[0, 4],4), len(values), round(values[0],4), round(values[1],4), round(values[2],4), round(values[3],4), round(values[4],4), round(values[5],4), round(values[6],4), round(values[7],4), round(values[8],4)

def main():
    output_lines = []
    
    if len(sys.argv) == 2:
        # Read values from the provided file
        file_path = sys.argv[1]
        data = np.loadtxt(file_path, delimiter=' ', skiprows=1, usecols=range(4, 13)) #Use 4, 14 for the file output by Paradiseo
        
        #Add header
        header = "Predicted_Density_[Kg/m^3] StdDeviation_Density Apparent_Closed_Porosity[/] StdDeviation_Porosity num_components waterglasscontent_[0,30] N330_[0.0,1.0] K3PO4_[0,4] Mn3O4_[0,7] drying_[NO,YES] mixing_[CLASSICAL,ADDITIONAL] furnace_temperature_[700,805] heating_rate_[1.0,5.0] foaming_time_[5.0,60]"
        output_lines.append("".join(map(str, header)))
        
        # Process each row in the file
        for row in data:
            prediction = process_values(row, "mlp_v1")
            output_lines.append(" ".join(map(str, prediction)))
            
    elif len(sys.argv) == 10:
        # Read values from the command line arguments
        waterglasscontent = float(sys.argv[1])
        N330 = float(sys.argv[2])
        K3PO4 = float(sys.argv[3])
        Mn3O4 = float(sys.argv[4])
        drying = str(sys.argv[5])  # NO , YES
        mixing = str(sys.argv[6])  # CLASSICAL, ADDITIONAL
        furnace_temperature = float(sys.argv[7])
        heating_rate = float(sys.argv[8])
        foaming_time = float(sys.argv[9])

        if drying == 'NO':
            drying = 0
        elif drying == 'YES':
            drying = 1
        else:
            drying = 0 if float(drying) < 0.5 else 1

        if mixing == 'CLASSICAL':
            mixing = 0
        elif mixing == 'ADDITIONAL':
            mixing = 1
        else:
            mixing = 0 if float(mixing) < 0.5 else 1

        myx = [waterglasscontent, N330, K3PO4, Mn3O4, drying, mixing, furnace_temperature, heating_rate, foaming_time]
        prediction = process_values(myx, "mlp_v1")
        output_lines.append(" ".join(map(str, prediction)))
    else:
        print("Invalid number of arguments. Provide either a file path or the required 9 values.")
        return
        
    # Write output to the file
    with open('IBEA.results', 'w') as output_file:
        for line in output_lines:
            output_file.write(line + "\n")

if __name__ == "__main__":
    main()
    
    
