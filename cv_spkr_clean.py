
import os
import shutil
import pandas as pd
from pydub import AudioSegment
from tqdm import tqdm
import subprocess

lang = "sl"      # language code
tsv_file_path = './current_version/sl_chr_spkr17.tsv'           # Replace with your TSV file path

threshold = 0.22                           # Set the threshold score here
action = "move_folders"                   # Choose "move_folders" to move the entire folder or "remove_files" to delete specific files




########### converting to wav and making the folders

def organize_and_convert_files(tsv_file_path, source_directory, destination_directory):
    # Load the TSV file into a DataFrame
    df = pd.read_csv(tsv_file_path, sep='\t')
    
    # Ensure destination directory exists
    if not os.path.exists(destination_directory):
        os.makedirs(destination_directory)
    
    # Get total number of files to process for the progress bar
    total_files = len(df)
    print(f"Starting file conversion and foldering for {total_files} files...")

    # Loop through each row in the DataFrame with progress bar
    for _, row in tqdm(df.iterrows(), total=total_files, desc="Converting files"):
        file_name = row['path']
        speaker_id = row['speaker_id']
        
        # Source path of the MP3 file
        source_path = os.path.join(source_directory, file_name)
        
        # Destination directory for the speaker
        speaker_folder = os.path.join(destination_directory, str(speaker_id))
        
        # Create the speaker folder if it doesn't exist
        if not os.path.exists(speaker_folder):
            os.makedirs(speaker_folder)
        
        # Destination path of the converted file
        destination_file_path = os.path.join(speaker_folder, file_name.replace('.mp3', '.wav'))
        
        # Convert the file to WAV with 16 kHz sample rate and save it in place
        if os.path.exists(source_path):
            # Load the MP3 file
            audio = AudioSegment.from_mp3(source_path)
            
            # Set the sample rate to 16 kHz
            audio = audio.set_frame_rate(16000)
            
            # Export as WAV and save it in the destination path
            audio.export(destination_file_path, format='wav')
            
            # Print conversion info
            # print(f"Converted and moved {file_name} to {speaker_folder} as WAV with 16 kHz")
            
            # Optionally, delete the original MP3 file from source
            os.remove(source_path)
        else:
            print(f"File {file_name} not found in source directory.")

    print("File conversion and organization completed!")

def create_trials_and_wav_scp(dataset_dir, trials_file, wav_scp_file):
    # Ensure output folder exists
    os.makedirs(os.path.dirname(trials_file), exist_ok=True)
    
    print("Creating trials.txt and wav.scp files...")

    # Create trials.txt and wav.scp files
    with open(trials_file, 'w') as trials, open(wav_scp_file, 'w') as wav_scp:
        # Process each speaker's folder in the dataset directory
        for speaker_folder in os.listdir(dataset_dir):
            speaker_path = os.path.join(dataset_dir, speaker_folder)
            
            # Ensure the path is a directory
            if os.path.isdir(speaker_path):
                # List all .wav files in the speaker's folder
                wav_files = sorted([f for f in os.listdir(speaker_path) if f.endswith('.wav')])
                
                # Ignore folders with only one file
                if len(wav_files) > 1:
                    # First file is the enrollment file for this speaker
                    enrollment_file = wav_files[0]
                    enrollment_file_path = os.path.join(speaker_path, enrollment_file)
                    
                    # Write enrollment file to wav.scp
                    wav_scp.write(f"{enrollment_file} {enrollment_file_path}\n")
                    
                    # Create trials for each remaining file in the folder
                    for trial_file in wav_files[1:]:
                        trial_file_path = os.path.join(speaker_path, trial_file)
                        
                        # Write each trial file to wav.scp
                        wav_scp.write(f"{trial_file} {trial_file_path}\n")
                        
                        # Write to trials.txt with label 1, assuming same speaker
                        trials.write(f"{trial_file} {enrollment_file} 1\n")
                    
                    # print(f"Processed speaker folder: {speaker_folder}")

    print("Trials and wav.scp files created successfully!")

# Define the parameters
# dataset_dir = f"./cv-corpus-17.0-2024-03-15/{lang}/clips/evaluated"
output_folder = f"./data/{lang}"
trials_file = os.path.join(output_folder, "trials")
wav_scp_file = os.path.join(output_folder, "wav.scp")


source_directory = f"./cv-corpus-17.0-2024-03-15/{lang}/clips"  # Replace with source directory
destination_directory = f"./cv-corpus-17.0-2024-03-15/{lang}/clips/evaluated"  # Replace with destination directory

# Run both functions
organize_and_convert_files(tsv_file_path, source_directory, destination_directory)


######### creating "wav.scp" and "trials" files

create_trials_and_wav_scp(destination_directory, trials_file, wav_scp_file)


######### extract mebedding feature for each audio file and making score file

subprocess.run(["bash", "run_eval_ver6.sh", lang])



######## manipulation of the files


shutil.move("LR_lang.txt", f"./data/{lang}/LR_lang.txt")
score_file = f"./data/{lang}/LR_lang.txt"   # Path to the score file
wav_scp_path = f"./data/{lang}/wav.scp"             # Path to the wav.scp file


def load_wav_scp(wav_scp_path):
    # Load the wav.scp file and create a mapping of file name to full path
    file_paths = {}
    with open(wav_scp_path, 'r') as f:
        for line in f:
            file_name, full_path = line.strip().split(maxsplit=1)
            file_paths[file_name] = full_path
    return file_paths

def determine_unclean_directory(wav_scp_path, levels_up=3):
    # Get the path of the first file in wav.scp to determine the base location
    with open(wav_scp_path, 'r') as f:
        first_line = f.readline().strip()
        _, first_file_path = first_line.split(maxsplit=1)
    
    # Go up the specified number of levels
    unclean_base_dir = first_file_path
    for _ in range(levels_up):
        unclean_base_dir = os.path.dirname(unclean_base_dir)
    
    # Define the unclean directory within the determined base directory
    unclean_dir = os.path.join(unclean_base_dir, "unclean")
    os.makedirs(unclean_dir, exist_ok=True)
    return unclean_dir

def move_or_remove_unclean_files(score_file, wav_scp_path, threshold, action):
    # Load wav.scp mapping to get full paths
    file_paths = load_wav_scp(wav_scp_path)
    
    # Determine the unclean directory if the action is "move_folders"
    unclean_dir = None
    if action == "move_folders":
        unclean_dir = determine_unclean_directory(wav_scp_path, levels_up=3)
    
    # Track speaker folders with scores below the threshold
    unclean_folders = {}
    
    # Read the score file and check for scores below the threshold
    with open(score_file, 'r') as file:
        for line in file:
            trial_file, enrollment_file, score = line.strip().split()
            score = float(score)
            
            # If the score is below the threshold, identify the speaker folder
            if score < threshold:
                # Get the full path for the trial file using wav.scp mapping
                trial_path = file_paths.get(trial_file)
                enrollment_path = file_paths.get(enrollment_file)
                if trial_path and enrollment_path:
                    # Extract the speaker folder (assuming it's the last directory in the path)
                    trial_folder = os.path.dirname(trial_path)
                    
                    # Add the trial pair and score to the unclean_folders dictionary
                    if trial_folder not in unclean_folders:
                        unclean_folders[trial_folder] = []
                    unclean_folders[trial_folder].append((trial_file, enrollment_file, score, trial_path))
                else:
                    print(f"Warning: {trial_file} or {enrollment_file} not found in wav.scp.")
    
    # Initialize counters
    moved_folders_count = 0
    removed_files_count = 0

    # Process the unclean files based on the chosen action
    for folder_path, low_score_pairs in unclean_folders.items():
        if os.path.isdir(folder_path):
            if action == "move_folders":
                # Move the entire folder to the unclean directory
                target_path = os.path.join(unclean_dir, os.path.basename(folder_path))
                shutil.move(folder_path, target_path)
                moved_folders_count += 1
                print(f"Moved '{folder_path}' to '{unclean_dir}'")
                
                # Create a log file listing low-score pairs in the moved unclean folder
                log_file_path = os.path.join(target_path, "low_score_pairs.txt")
                with open(log_file_path, 'w') as log_file:
                    log_file.write("Trial File\tEnrollment File\tScore\n")
                    for trial_file, enrollment_file, score, _ in low_score_pairs:
                        log_file.write(f"{trial_file}\t{enrollment_file}\t{score:.4f}\n")
                
                print(f"Created 'low_score_pairs.txt' in '{target_path}'")
            
            elif action == "remove_files":
                # Only remove the specific trial files with low scores in the current folder
                for trial_file, enrollment_file, score, trial_path in low_score_pairs:
                    if os.path.exists(trial_path):
                        os.remove(trial_path)
                        removed_files_count += 1
                        print(f"Removed trial file: {trial_path}")
                    
                # Create a log file listing low-score pairs in the unclean directory
                log_file_path = os.path.join(folder_path, "low_score_pairs.txt")
                with open(log_file_path, 'w') as log_file:
                    log_file.write("Trial File\tEnrollment File\tScore\n")
                    for trial_file, enrollment_file, score, _ in low_score_pairs:
                        log_file.write(f"{trial_file}\t{enrollment_file}\t{score:.4f}\n")
                
                print(f"Created 'low_score_pairs.txt' in '{folder_path}'")

    # Final report
    if action == "move_folders":
        print(f"\nTotal folders moved: {moved_folders_count}")
    elif action == "remove_files":
        print(f"\nTotal trial files removed: {removed_files_count}")

    print("All files processed successfully.")

# Run the function with specified parameters
move_or_remove_unclean_files(score_file, wav_scp_path, threshold, action)



