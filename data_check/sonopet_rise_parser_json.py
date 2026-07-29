# read and analyse data from Sonopet and Core data generated from RISE tower
# py Sonopet_RISE_parser.py SonopetCase_2025_08_12_T_10_33_00_2506710269.json 
# py Sonopet_RISE_parser.py SonopetCase_2025_08_12_T_14_21_00_2330410169.json
# py Sonopet_RISE_parser.py SonopetCase_2025_08_14_T_15_48_00_2506710269.json 



import json
import pandas as pd
from pathlib import Path
import sys

def load_records(json_path):
    """Load either a JSON array file or a newline-delimited Sonopet JSON stream."""
    with open(json_path, "r") as f:
        first = f.read(1)
        f.seek(0)
        if first == "[":
            return json.load(f)
        return [json.loads(line) for line in f if line.strip()]


def parse_log(json_path):
        
    # Create the output file name
    data = load_records(json_path)
    output_csv = Path(json_path).with_suffix('')  # remove .json extension
    runtime_output_csv = str(output_csv) + "_runtime_data.csv"        
    generic_info_output_csv = str(output_csv) + "_generic_info.csv"   
    device_errors_output_csv = str(output_csv) + "_device_errors.csv"         
    device_connections_output_csv = str(output_csv) + "_device_connections.csv" 
          
    # Prepare empty list for generic info
    generic_info_list = []
 
    for entry in data:
        generic_info = entry.get("Generic_Info", {})
        if generic_info and generic_info not in generic_info_list:
            generic_info_list.append(generic_info)
    df_generic = pd.DataFrame(generic_info_list)
    df_generic.to_csv(generic_info_output_csv, index=False)
    print("✅ Saved generic_info.csv")

    # Prepare empty lists for other outputs
    connections_rows = []
    errors_rows = []
    runtime_rows = []

    for entry in data:
        if "timestamp" in entry and isinstance(entry.get("data"), dict):
            runtime_rows.append({"timestamp": entry["timestamp"], **entry["data"]})
            continue
 
        # Device Connection
        if entry.get("Device_Connection"):
            conn_data = entry["Device_Connection"]
            if isinstance(conn_data, dict):
                conn_row = {**generic_info, **conn_data}
                connections_rows.append(conn_row)
 
        # Device Error
        if entry.get("Device_Error"):
            err_data = entry["Device_Error"]
            if isinstance(err_data, dict):
                err_row = {**generic_info, **err_data}
                errors_rows.append(err_row)
 
        # Runtime Info (can be a list of dicts)
        if entry.get("Ultrasonic_Runtime_Info_Array"):
            runtime_data = entry["Ultrasonic_Runtime_Info_Array"]
            if isinstance(runtime_data, dict):
                runtime_data = [runtime_data]
            for r in runtime_data:
                runtime_row = {**generic_info, **r}
                runtime_rows.append(runtime_row)
 
    # Convert to DataFrames
    df_connections = pd.DataFrame(connections_rows)
    df_errors = pd.DataFrame(errors_rows)
    df_runtime = pd.DataFrame(runtime_rows)
 
    # Save to CSV
    df_connections.to_csv(device_connections_output_csv, index=False)
    df_errors.to_csv(device_errors_output_csv, index=False)
    df_runtime.to_csv(runtime_output_csv, index=False)
 
    print("✅ Saved device_connections.csv, device_errors.csv, runtime_data.csv")
    return df_runtime


if __name__ == "__main__":
    folder ="/media/btllab/B2EEF271EEF22CEB/Ubuntu/franka3-sonopet-ros2/data_collection/experiments/20260601T155242_chicken_1_90_50_15_3/sonopet"
    json_files = sorted(Path(folder).glob("SonopetCase_*.json"))
    runtime_tables = []
    for json_file in json_files:
        df_runtime = parse_log(json_file)
        if not df_runtime.empty:
            df_runtime = df_runtime.copy()
            df_runtime.insert(0, "source_json", json_file.name)
            runtime_tables.append(df_runtime)

    if len(runtime_tables) > 1:
        combined = pd.concat(runtime_tables, ignore_index=True, sort=False)
        if "timestamp" in combined.columns:
            combined = combined.sort_values("timestamp").reset_index(drop=True)
        combined.to_csv(Path(folder) / "combined_runtime_data.csv", index=False)
        print("✅ Saved combined_runtime_data.csv")

