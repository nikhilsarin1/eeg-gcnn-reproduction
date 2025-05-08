"""
Helper functions to download EEG datasets from GitHub.
"""

import os
import subprocess
import shutil
import tempfile
import platform
import sys

def download_from_github(github_url, output_dir, depth=1, branch='main'):
    """
    Download a repository from GitHub using git clone.
    
    Args:
        github_url (str): URL to the GitHub repository.
        output_dir (str): Directory to save the downloaded data.
        depth (int): Git clone depth (how many commits to download).
        branch (str): Branch to clone.
        
    Returns:
        bool: True if download was successful, False otherwise.
    """
    print(f"Downloading from {github_url} to {output_dir}...")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Create a temporary directory for the clone
    with tempfile.TemporaryDirectory() as temp_dir:
        try:
            # Check if git is installed
            try:
                subprocess.run(['git', '--version'], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            except (subprocess.SubprocessError, FileNotFoundError):
                print("Error: Git is not installed or not in PATH.")
                print("Please install Git to download from GitHub.")
                return False
            
            # Clone the repository
            clone_cmd = ['git', 'clone', '--depth', str(depth), '--branch', branch, github_url, temp_dir]
            subprocess.run(clone_cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            # Move files from temp directory to output directory
            for item in os.listdir(temp_dir):
                # Skip .git directory
                if item == '.git':
                    continue
                
                source = os.path.join(temp_dir, item)
                destination = os.path.join(output_dir, item)
                
                if os.path.isdir(source):
                    # If directory already exists, merge contents
                    if os.path.exists(destination):
                        # Copy contents
                        for sub_item in os.listdir(source):
                            sub_source = os.path.join(source, sub_item)
                            sub_destination = os.path.join(destination, sub_item)
                            if os.path.isdir(sub_source):
                                shutil.copytree(sub_source, sub_destination, dirs_exist_ok=True)
                            else:
                                shutil.copy2(sub_source, sub_destination)
                    else:
                        # Copy entire directory
                        shutil.copytree(source, destination)
                else:
                    # Copy file
                    shutil.copy2(source, destination)
            
            print(f"Download completed successfully to {output_dir}")
            return True
            
        except Exception as e:
            print(f"Error downloading from GitHub: {e}")
            return False

def download_lemon_dataset(output_dir, github_url='https://github.com/OpenNeuroDatasets/ds000221'):
    """
    Download the MPI LEMON dataset from GitHub.
    
    Args:
        output_dir (str): Directory to save the downloaded data.
        github_url (str): URL to the GitHub repository.
        
    Returns:
        bool: True if download was successful, False otherwise.
    """
    print("Downloading MPI LEMON dataset from GitHub...")
    
    # Create output directory
    lemon_dir = os.path.join(output_dir, 'lemon')
    os.makedirs(lemon_dir, exist_ok=True)
    
    # Download from GitHub
    success = download_from_github(github_url, lemon_dir)
    
    if success:
        print(f"MPI LEMON dataset downloaded to {lemon_dir}")
        return True
    else:
        print("Failed to download MPI LEMON dataset.")
        print("You can manually download it from:", github_url)
        return False

def download_eeg_gcnn_precomputed(output_dir):
    """
    Download precomputed features for EEG-GCNN reproduction from FigShare.
    
    Args:
        output_dir (str): Directory to save the downloaded files
        
    Returns:
        bool: True if download was successful, False otherwise
    """
    import requests
    import zipfile
    import os
    from tqdm import tqdm
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # FigShare direct download link
    figshare_url = "https://ndownloader.figshare.com/files/37681734"
    zip_path = os.path.join(output_dir, "eeg_gcnn_precomputed.zip")
    
    print(f"Downloading precomputed features from FigShare...")
    try:
        # Download with progress bar
        response = requests.get(figshare_url, stream=True)
        total_size = int(response.headers.get('content-length', 0))
        block_size = 1024
        
        with open(zip_path, 'wb') as f, tqdm(
                desc="Downloading",
                total=total_size,
                unit='B',
                unit_scale=True,
                unit_divisor=1024,
            ) as pbar:
            for data in response.iter_content(block_size):
                f.write(data)
                pbar.update(len(data))
        
        # Extract zip file
        print(f"Extracting files to {output_dir}...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(output_dir)
        
        # Remove zip file after extraction
        os.remove(zip_path)
        print(f"Download and extraction complete!")
        return True
    
    except Exception as e:
        print(f"Error downloading precomputed features: {e}")
        return False

if __name__ == '__main__':
    # This allows running this script directly to download the dataset
    download_lemon_dataset('data/raw')