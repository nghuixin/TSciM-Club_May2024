import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
import re
import argparse  # For parsing command-line arguments.
import os
import pre_process # Custom module from pre_process.py
from monai.networks.nets import DenseNet   # Importing the DenseNet model from MONAI.
import torch  # PyTorch library for deep learning operations
import nibabel as nib # For handling neuroimaging data.
import tqdm # Display progress bar
from collections import OrderedDict
from captum.attr import GuidedBackprop

# Script loads trained model weights, performs inference on input MRI data specified in a CSV file, 
# and optionally evaluates the model's predictions against provided ages. 

# Function to adjust the model state dictionary (useful when switching from parallel to single-device training).

def convert_state_dict(input_path):
    # function to remove the keywork 'module' from pytorch state_dict (which occurs when model is trained using nn.DataParallel)
    new_state_dict = OrderedDict()
    state_dict = torch.load(input_path, map_location='cpu')
    for k, v in state_dict.items():
        if 'module' in k:
            name = k[7:] # remove `module.`
        else:
            name = k
        new_state_dict[name] = v
    return new_state_dict
           
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', dest='gpu', action='store_true')
    parser.set_defaults(gpu=False)
    parser.add_argument('--return_metrics', dest='return_metrics', action='store_true')
    parser.set_defaults(return_metrics=False)
    parser.add_argument('--skull_strip', dest='skull_strip', action='store_true')
    parser.set_defaults(skull_strip=False)
    parser.add_argument('--pred_correction', dest='pred_correction', action='store_true')
    parser.set_defaults(pred_correction=False)
    parser.add_argument('--ensemble', dest='ensemble', action='store_true')
    parser.set_defaults(ensemble=False)
    parser.add_argument('--sequence', type=str, default='t2')
    parser.add_argument('--csv_file', type=str, required=True)
    parser.add_argument('--project_name', type=str, required=True)

    args = parser.parse_args()
    if not os.path.exists('./{}'.format(args.project_name)):
        os.mkdir('./{}'.format(args.project_name))
    # else: # commented out so it will always overwrite the existing folder of the same name
    #    raise ValueError('project name {} aready used'.format(args.project_name))
    if args.gpu:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')     
    
    if args.sequence == 't2':
        if args.skull_strip:
            state_dict = convert_state_dict('./Models/T2/Skull_stripped/seed_42.pt')
            net = DenseNet(3,1,1)
            net.load_state_dict(state_dict)
            net = net.to(device)
            net.eval()
        else:
            state_dict = convert_state_dict('./Models/T2/Raw/seed_42.pt')
            net = DenseNet(3,1,1)
            net.load_state_dict(state_dict)
            net = net.to(device)
            net.eval()
    elif args.sequence == 't1':
        if args.skull_strip:
            if args.ensemble:
                net = []
                for path in os.listdir('./Models/T1/Skull_stripped/'):
                    state_dict = convert_state_dict('./Models/T1/Skull_stripped/' + path)
                    Net = DenseNet(3,1,1)
                    Net.load_state_dict(state_dict)
                    Net = Net.to(device)
                    Net.eval()
                    net.append(Net)
            else:
                state_dict = convert_state_dict('./Models/T1/Skull_stripped/seed_60.pt')
                net = DenseNet(3,1,1)
                net.load_state_dict(state_dict)
                net = net.to(device)
                net.eval()        
        else:
            raise ValueError('Raw T1 model not currently handled. Please specify --skull_strip if skull-stripped and registered (MNI152) T1 model is desired')
    else:
        raise ValueError('{} MRI sequence not currently handled (must be one of t2 or t1; DWI and FLAIR coming soon!)'.format(args.sequence))

    df = pd.read_csv(args.csv_file)
    
    assert args.sequence in ['t1','t2'], '''Unsupported sequence provided ({})'''.format(args.sequence)
    
    assert 'file_name' in df.columns, '''No column named 'file_name' in csv_file'''
    
    assert 'ID' in df.columns, '''No column named 'ID' in csv_file'''
    
    if args.return_metrics:
        assert 'Age' in df.columns, '''No column named 'Age' in csv_file, can't return brain-age metrics (MAE, Pearson's etc.)'''  
        
    if args.pred_correction:
        assert 'Age' in df.columns, '''No column named 'Age' in csv_file, can't correct for bias in brain-age predictions'''

# ------------------------------------
# Processing loop for each MRI file listed in the CSV
    brain_predicted_ages = [] 
    chronological_ages = []
    IDs = []
    average_attributions = []  # Initialize list to store average attributions


    # Evaluation loop
    with torch.no_grad(): # disable gradient calculations cos it is now in inference phase
        # inference phase = model used for predictions not for training
        for index, row in tqdm.tqdm(df.iterrows(), total=df.shape[0]): # iterates over each row in df from csv file
            # tqdm used to add progress bar to monitor how many images have been processed
            file_name = row['file_name'] 
            ID = row['ID'] 
            if args.return_metrics: # check if user requested to return loss function like MAE
                age = row['Age']
            
            # use custom module to preprocess the MRI input
            processed_arr = pre_process.preprocess(input_path=file_name, use_gpu=args.gpu, skull_strip=args.skull_strip, register=args.sequence=='t1', project_name=args.project_name)
            print("Pre-processing completed.")

            if not type(processed_arr)==np.ndarray:
                continue # Skip current iteration if pre processing fails

            #mri_voxel_sample = processed_arr[0, 13, 42, 57] # Extract a specific voxel
            #mri_slice_sag = processed_arr[0, 60 , : , :] # Extract a specific slice along the axial plane
            #mri_slice_cor = processed_arr[0, : , 60 , :] 
            mri_slice_ax = processed_arr[0, : , : , 60] 

            tensor = torch.from_numpy(processed_arr).view(1,1,130,130,130) # converts np arr into PyTorch tensor
            # plt.figure(figsize=(6, 6))  # Set the figure size
            # plt.imshow(mri_slice_sag, cmap='gray')  # Plot the slice with a grayscale colormap
            # plt.title(f'Sagittal Slice for ID {ID}')  # Add a title to the plot
            # plt.axis('off')  # Turn off axis numbers and ticks
            # plt.savefig(f'./{args.project_name}/{ID}_slice_sag.png')  # Save the figure to a file
            #plt.close()  # Close the plot figure to free memory
            
            # plt.figure(figsize=(6, 6))
            # plt.imshow(mri_slice_cor, cmap='gray')  # Plot the slice with a grayscale colormap
            # plt.title(f'Coronal Slice for ID {ID}')  # Add a title to the plot
            # plt.axis('off')  # Turn off axis numbers and ticks
            # plt.savefig(f'./{args.project_name}/{ID}_slice_coro.png')  # Save the figure to a file
            # plt.close()  # Close the plot figure to free memory

            #plt.figure(figsize=(6, 6))
            #plt.imshow(mri_slice_ax, cmap='gray')  # Plot the slice with a grayscale colormap
            #plt.title(f'Axial Slice for ID {ID}')  # Add a title to the plot
            #plt.axis('off')  # Turn off axis numbers and ticks
            #plt.savefig(f'./{args.project_name}/{ID}_slice_ax.png')  # Save the figure to a file
            #plt.close()  # Close the plot figure to free memory

            tensor = (tensor - tensor.mean())/tensor.std() # Normalize the tensor
            tensor = torch.clamp(tensor,-1,5) # clamp values to remove outliers
            # changes data type of tensor to float
            tensor = tensor.to(device=device, dtype = torch.float)
            
            # ---- Interpretability Method ----- 
            gbp = GuidedBackprop(net) # captum 
            attribution = gbp.attribute(tensor) # get the attribution map of the inputs using gbp
            # np.save('attribution.npy', attribution) 

            # Selecting the axial slice - ensure the indices match the dimensions of your tensor
            attribution_np = attribution.detach().cpu().numpy()
            # = attribution_np[0, 0, :, :, 60]  # adjust the indices based on how your tensor is structured
            #slice_count = len(slice_numbers)  # Number of slices
            #average_saliencies = []  # List to store average saliency values
            attribution_slice = attribution_np[0, 0, :, :, 60]
            average_attributions.append(attribution_slice)

            # # Loop through each slice index
            # for i in slice_numbers: # implement tqdm
            #     attribution_slice = attribution_np[0, 0, :, :, i]
            #     average_saliency = np.mean(attribution_slice)
            #     average_saliencies.append(average_saliency)  # Append average saliency to the list

            # Plotting
            #plt.figure(figsize=(10, 6))
            # plt.plot(slice_numbers, average_saliencies, marker='o')  # Plot slice number vs. average saliency
            # plt.xlabel('Slice Number')
            # plt.ylabel('Average Saliency')
            # plt.title('Average Saliency per Slice')
            # plt.grid(True)

            # # Customizing x and y ticks
            # plt.xticks(slice_numbers)  # Set x ticks to match the slice numbers
            # plt.yticks()  # You can set custom y ticks if needed

            # plt.savefig(f'./{args.project_name}/saliency_distribution.png')  # Save the figure to a file
            # plt.close()  # Close the plot figure to free memory
            
            # Normalize the slice for better visualization
            normalized_attribution  = (attribution_slice - np.min(attribution_slice)) / (np.max(attribution_slice) - np.min(attribution_slice))

            # # Display the MRI slice
            plt.subplot(1, 2, 1)
            plt.imshow(mri_slice_ax, cmap='gray')
            plt.title('Original MRI Axial Slice')
            plt.axis('off')

            # Display the overlay of the attribution on the MRI slice

            # plt.axis('off')
            # plt.savefig(f'./{args.project_name}/{ID}_slice_attribution.png')  # Save the figure to a file
            # plt.close()  # Close the plot figure to free memory
            
            
            if args.sequence=='t1':
                if args.ensemble:
                    temp_preds = []
                    for Net in net:
                        temp_preds.append(np.round(Net(tensor).detach().cpu().item(), 1))
                    brain_predicted_ages.append(np.mean(temp_preds))
                else:
                     brain_predicted_ages.append(np.round(net(tensor).detach().cpu().item(), 1))
            else:
                if args.pred_correction and not args.skull_strip:              
                    brain_predicted_ages.append(np.round(net(tensor).detach().cpu().item(), 1) - (-0.0627*age + 2.54))
                elif args.pred_correction and args.skull_strip:              
                    brain_predicted_ages.append(np.round(net(tensor).detach().cpu().item(), 1) - (-0.0854*age + 2.67))
                else:
                    brain_predicted_ages.append(np.round(net(tensor).detach().cpu().item(), 1))
            if args.return_metrics:
                chronological_ages.append(np.round(row['Age'],1))
            IDs.append(ID)
            
    mean_attribution = np.mean(average_attributions, axis=0)
    plt.subplot(1, 2, 2)
    plt.imshow(mri_slice_ax, cmap='gray')  # MRI slice in grayscale
    plt.imshow(mean_attribution, cmap='jet', alpha=0.5)  # Attribution overlay with transparency
    plt.title('Attribution Overlay on MRI Axial Slice')
    plt.axis('off')
    plt.savefig(f'./{args.project_name}/average_axial_attribution.png')  # Save the figure to a file

    if args.return_metrics:
        pd.DataFrame({'ID':IDs,'Chronological age':chronological_ages,'Predicted_age (years)':brain_predicted_ages}).set_index('ID').to_csv('./{}_brain_age_output.csv'.format(args.project_name))
        MAE = sum([np.abs(a-b) for a, b in zip(brain_predicted_ages, chronological_ages)])/len(brain_predicted_ages)
        corr_mat = np.corrcoef(chronological_ages, brain_predicted_ages)
        corr = corr_mat[0,1]
        fig = plt.figure(figsize=(8,8))
        ax = fig.add_subplot(111)
        ax.scatter(chronological_ages, brain_predicted_ages, alpha=0.3)
        ax.plot(chronological_ages, chronological_ages,linestyle= '--', color='black')
        ax.set_ylim([min(chronological_ages), max(chronological_ages)])
        ax.set_aspect('equal')
        ax.set_xlabel('Chronological age')
        ax.set_ylabel('Predicted age')
        ax.set_title('MAE = {:.2f} years, p = {:.2f}\n'.format(MAE, corr))
        fig.savefig('./{}/scatter.png'.format(args.project_name), facecolor='w')
    else:
        pd.DataFrame({'ID':IDs,'Predicted_age (years)':brain_predicted_ages}).set_index('ID').to_csv('./{}/brain_age_output.csv'.format(args.project_name)) 
