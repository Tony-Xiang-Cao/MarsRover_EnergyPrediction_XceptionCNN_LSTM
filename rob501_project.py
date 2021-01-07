from pathlib import Path
import argparse
import sys
from support.test_docker import test_docker
#----- ADD YOUR IMPORTS HERE IF NEEDED -----
import os
import Pre_process as pre
import numpy as np


def run_project(input_dir, output_dir):
    """
    Main entry point for your project code. 
    
    DO NOT MODIFY THE SIGNATURE OF THIS FUNCTION.
    """
    #---- FILL ME IN ----
       
    
    #Define the path of dataset in input directory
    #Run 2 is the holdout testing dataset
    run2_img_dir = os.path.join(input_dir, 'run2_base_hr/omni_image5')
    run2_dir = os.path.join(input_dir, 'run2_base_hr')
    
    #load training image and power data
    run2_img= pre.load_image(run2_img_dir)
    run2_power = pre.load_power(run2_dir,12,2170)
    
    #load pre-trained model and predict on testset
    saved_model = pre.load_model()
    predictions = np.squeeze(saved_model.predict(run2_img, batch_size=32)[:,0,0])
 
    mae = np.mean(np.absolute(run2_power - predictions))
    print("The MAE evaluated on testset Run2 is : ", mae)
    
    mean_power_2 = np.mean(run2_power)
    print("The MAE value is 24.5% of the aveage power ", mean_power_2, "Watt")
    pass

    #--------------------


# Command Line Arguments
parser = argparse.ArgumentParser(description='ROB501 Final Project.')
parser.add_argument('--input_dir', dest='input_dir', type=str, default="./input",
                    help='Input Directory that contains all required rover data')
parser.add_argument('--output_dir', dest='output_dir', type=str, default="./output",
                    help='Output directory where all outputs will be stored.')


if __name__ == "__main__":
    
    # Parse command line arguments
    args = parser.parse_args()

    # Uncomment this line if you wish to test your docker setup
    #test_docker(Path(args.input_dir), Path(args.output_dir))

    # Run the project code
    run_project(args.input_dir, args.output_dir)