import os
import numpy as np
from typing import List, Tuple

def save_epsilon_machine_to_file(epsilon_machine: np.ndarray, num_states: int, repeat_index: int) -> str:
    """Save the epsilon machine to a file and return the filename."""
    filename = f"data/epsilon_machine_{num_states}_{repeat_index}.npz"
    np.savez(filename, epsilon_machine=epsilon_machine)
    return filename


def save_model_to_drive(model, drive, epoch, sweep_name):
    # Get or create the folder with sweep name
    folder_id = create_or_get_drive_folder(drive, sweep_name)

    # Specify the filename and path
    filename = f'model_epoch_{epoch}.pt'
    path = f'./{filename}'

    # Save the model state
    torch.save(model.state_dict(), path)

    # Create & upload a file to Google Drive
    file = drive.CreateFile({'title': filename, 'parents': [{'id': folder_id}]})
    file.SetContentFile(path)
    file.Upload()

    # Remove the local file to save space
    os.remove(path)

def create_or_get_drive_folder(drive, folder_name):
    # Search for the folder
    folder_list = drive.ListFile({'q': f"title='{folder_name}' and mimeType='application/vnd.google-apps.folder' and trashed=false"}).GetList()
    
    if folder_list:
        # Folder exists, return its ID
        folder_id = folder_list[0]['id']
    else:
        # Folder doesn't exist, create it
        folder_metadata = {'title': folder_name, 'mimeType': 'application/vnd.google-apps.folder'}
        folder = drive.CreateFile(folder_metadata)
        folder.Upload()
        folder_id = folder['id']
    
    return folder_id