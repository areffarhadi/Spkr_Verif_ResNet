import os
import shutil
import pandas as pd
from pydub import AudioSegment
from tqdm import tqdm
import subprocess

# Input file containing language codes and TSV file paths
input_file = 'lang_paths.txt'  # This file should contain lines like "hi ./current_version/hi_epi_spkr17.tsv"

# Global parameters
threshold = 0.3                          # Set the threshold score here
action = "move_folders"                  # Choose "move_folders" to move the entire folder or "remove_files" to delete specific files

def organize_and_convert_files(tsv_file_path, source_directory, destination_directory):
    df = pd.read_csv(tsv_file_path, sep='\t')
    os.makedirs(destination_directory, exist_ok=True)
    
    print(f"Starting file conversion and organization for {len(df)} files...")
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Converting files"):
        file_name, speaker_id = row['path'], row['speaker_id']
        source_path = os.path.join(source_directory, file_name)
        speaker_folder = os.path.join(destination_directory, str(speaker_id))
        os.makedirs(speaker_folder, exist_ok=True)
        destination_file_path = os.path.join(speaker_folder, file_name.replace('.mp3', '.wav'))
        
        if os.path.exists(source_path):
            audio = AudioSegment.from_mp3(source_path).set_frame_rate(16000)
            audio.export(destination_file_path, format='wav')
            os.remove(source_path)  # Optionally delete original MP3
        else:
            print(f"File {file_name} not found in source directory.")
    print("File conversion and organization completed!")

def create_trials_and_wav_scp(dataset_dir, trials_file, wav_scp_file):
    os.makedirs(os.path.dirname(trials_file), exist_ok=True)
    print("Creating trials.txt and wav.scp files...")

    with open(trials_file, 'w') as trials, open(wav_scp_file, 'w') as wav_scp:
        for speaker_folder in os.listdir(dataset_dir):
            speaker_path = os.path.join(dataset_dir, speaker_folder)
            if os.path.isdir(speaker_path):
                wav_files = sorted(f for f in os.listdir(speaker_path) if f.endswith('.wav'))
                if len(wav_files) > 1:
                    enrollment_file = wav_files[0]
                    enrollment_file_path = os.path.join(speaker_path, enrollment_file)
                    wav_scp.write(f"{enrollment_file} {enrollment_file_path}\n")
                    
                    for trial_file in wav_files[1:]:
                        trial_file_path = os.path.join(speaker_path, trial_file)
                        wav_scp.write(f"{trial_file} {trial_file_path}\n")
                        trials.write(f"{trial_file} {enrollment_file} 1\n")
    print("Trials and wav.scp files created successfully!")

def load_wav_scp(wav_scp_path):
    file_paths = {}
    with open(wav_scp_path, 'r') as f:
        for line in f:
            file_name, full_path = line.strip().split(maxsplit=1)
            file_paths[file_name] = full_path
    return file_paths

def determine_unclean_directory(wav_scp_path, levels_up=3):
    with open(wav_scp_path, 'r') as f:
        first_line = f.readline().strip()
        _, first_file_path = first_line.split(maxsplit=1)
    
    unclean_base_dir = first_file_path
    for _ in range(levels_up):
        unclean_base_dir = os.path.dirname(unclean_base_dir)
    
    unclean_dir = os.path.join(unclean_base_dir, "unclean")
    os.makedirs(unclean_dir, exist_ok=True)
    return unclean_dir

def move_or_remove_unclean_files(score_file, wav_scp_path, threshold, action):
    file_paths = load_wav_scp(wav_scp_path)
    
    unclean_dir = determine_unclean_directory(wav_scp_path, levels_up=3) if action == "move_folders" else None
    unclean_folders = {}

    with open(score_file, 'r') as file:
        for line in file:
            trial_file, enrollment_file, score = line.strip().split()
            score = float(score)
            if score < threshold:
                trial_path = file_paths.get(trial_file)
                enrollment_path = file_paths.get(enrollment_file)
                if trial_path and enrollment_path:
                    trial_folder = os.path.dirname(trial_path)
                    if trial_folder not in unclean_folders:
                        unclean_folders[trial_folder] = []
                    unclean_folders[trial_folder].append((trial_file, enrollment_file, score, trial_path))
    
    moved_folders_count, removed_files_count = 0, 0

    for folder_path, low_score_pairs in unclean_folders.items():
        if os.path.isdir(folder_path):
            if action == "move_folders":
                target_path = os.path.join(unclean_dir, os.path.basename(folder_path))
                shutil.move(folder_path, target_path)
                moved_folders_count += 1
                log_file_path = os.path.join(target_path, "low_score_pairs.txt")
                with open(log_file_path, 'w') as log_file:
                    log_file.write("Trial File\tEnrollment File\tScore\n")
                    for trial_file, enrollment_file, score, _ in low_score_pairs:
                        log_file.write(f"{trial_file}\t{enrollment_file}\t{score:.4f}\n")
                
            elif action == "remove_files":
                for trial_file, enrollment_file, score, trial_path in low_score_pairs:
                    if os.path.exists(trial_path):
                        os.remove(trial_path)
                        removed_files_count += 1
                    
                log_file_path = os.path.join(folder_path, "low_score_pairs.txt")
                with open(log_file_path, 'w') as log_file:
                    log_file.write("Trial File\tEnrollment File\tScore\n")
                    for trial_file, enrollment_file, score, _ in low_score_pairs:
                        log_file.write(f"{trial_file}\t{enrollment_file}\t{score:.4f}\n")

    if action == "move_folders":
        print(f"\nTotal folders moved: {moved_folders_count}")
    elif action == "remove_files":
        print(f"\nTotal trial files removed: {removed_files_count}")
    print("All files processed successfully.")

# Process each line in the input file
with open(input_file, 'r') as f:
    for line in f:
        lang, tsv_file_path = line.strip().split()
        source_directory = f"./cv-corpus-17.0-2024-03-15/{lang}/clips"
        destination_directory = f"./cv-corpus-17.0-2024-03-15/{lang}/clips/evaluated"
        output_folder = f"./data/{lang}"
        trials_file = os.path.join(output_folder, "trials")
        wav_scp_file = os.path.join(output_folder, "wav.scp")
        
        # Step 1: Convert files to wav and organize folders
        organize_and_convert_files(tsv_file_path, source_directory, destination_directory)
        
        # Step 2: Create trials.txt and wav.scp
        create_trials_and_wav_scp(destination_directory, trials_file, wav_scp_file)
        
        # Step 3: Run feature extraction script
        subprocess.run(["bash", "run_eval_ver6.sh", lang])

        # Step 4: Move score file and process unclean files
        shutil.move("LR_lang.txt", f"{output_folder}/LR_lang.txt")
        score_file = f"{output_folder}/LR_lang.txt"
        move_or_remove_unclean_files(score_file, wav_scp_file, threshold, action)

