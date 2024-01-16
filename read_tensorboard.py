from tensorboard.backend.event_processing import event_accumulator
import pandas as pd
import os
import glob

# Output Tensorboard logs into an Excel spreadsheet.
# Only the last step would be recorded.


def read_all_tensorboard_event_files(folder_path):
    all_data = []
    for class_folder in os.listdir(folder_path):
        class_path = os.path.join(folder_path, class_folder)
        for version_folder in os.listdir(class_path):
            version_path = os.path.join(class_path, version_folder)
            if os.path.isdir(version_path):
                event_file_path = os.path.join(version_path, 'events.out.tfevents.*')
                event_files = sorted(glob.glob(event_file_path), key=os.path.getmtime)

                if event_files:
                    last_event_file = event_files[-1]
                    data = read_tensorboard_event_file(last_event_file)
                    data['Class'] = class_folder.__str__()
                    all_data.append(data)

    return pd.DataFrame(all_data)


def read_tensorboard_event_file(event_file_path):
    event_acc = event_accumulator.EventAccumulator(event_file_path)
    event_acc.Reload()

    # Get all scalar tags
    scalar_tags = event_acc.Tags()['scalars']

    # Extract data for the last step for each scalar
    data = {}
    for tag in scalar_tags:
        events = event_acc.Scalars(tag)
        if events:
            last_event = events[-1]
            data[tag] = last_event.value
        else:
            data[tag] = None

    # Add a separate entry for 'step' and 'time_elapsed'
    data['Epochs'] = last_event.step
    data['Time Taken (s)'] = last_event.wall_time - events[0].wall_time

    return data


def write_to_excel(data, output_file):
    # Create a DataFrame from the data
    df = pd.DataFrame(data)

    # Write to Excel file
    df.to_excel(output_file, index=False)


if __name__ == "__main__":
    # Provide the path to the folder containing all version folders
    tensorboard_folder = "lightning_logs"

    # Provide the path where you want to save the Excel file
    excel_output_file = "output.xlsx"

    # Read all TensorBoard event files in the folder and merge results
    merged_data = read_all_tensorboard_event_files(tensorboard_folder)

    # Write merged data to Excel file
    write_to_excel(merged_data, excel_output_file)

    print(f"Data from all TensorBoard event files in {tensorboard_folder} has been merged and written to {excel_output_file}.")