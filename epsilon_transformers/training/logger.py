import csv
import json
import os
from typing import Dict, Any

class StructuredLogger:
    def __init__(self, base_dir: str, run_id: str):
        self.base_dir = os.path.join(base_dir, run_id)
        os.makedirs(self.base_dir, exist_ok=True)
        self.csv_file = open(os.path.join(self.base_dir, 'log.csv'), 'w', newline='')
        self.csv_writer = csv.writer(self.csv_file)
        self.json_file = open(os.path.join(self.base_dir, 'log.json'), 'w')

    def log(self, data: Dict[str, Any]):
        # Flatten the dictionary for CSV
        flat_data = self._flatten_dict(data)
        if not hasattr(self, 'csv_headers'):
            self.csv_headers = list(flat_data.keys())
            self.csv_writer.writerow(self.csv_headers)
        self.csv_writer.writerow([flat_data.get(h, '') for h in self.csv_headers])

        # Write to JSON
        json.dump(data, self.json_file)
        self.json_file.write('\n')
        self.json_file.flush()

    def _flatten_dict(self, d: Dict[str, Any], parent_key: str = '', sep: str = '.') -> Dict[str, Any]:
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)

    def close(self):
        self.csv_file.close()
        self.json_file.close()