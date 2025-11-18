"""
WandB Run Counter
Maintains persistent run counter for experiment tracking
"""

import os
import json


class WandBRunCounter:
    """
    Persistent counter for WandB runs
    Stores state in a JSON file to maintain count across sessions
    """

    def __init__(self, counter_file='logs/wandb_run_counter.json', project_name='MicroVLM-M'):
        self.counter_file = counter_file
        self.project_name = project_name

        # Ensure directory exists
        os.makedirs(os.path.dirname(counter_file), exist_ok=True)

        # Load or initialize counter
        self.counter_data = self._load_counter()

    def _load_counter(self):
        """Load counter from file or initialize"""
        if os.path.exists(self.counter_file):
            with open(self.counter_file, 'r') as f:
                return json.load(f)
        else:
            return {'count': 0, 'runs': []}

    def _save_counter(self):
        """Save counter to file"""
        with open(self.counter_file, 'w') as f:
            json.dump(self.counter_data, f, indent=2)

    def get_next_run_name(self, config_name=None):
        """
        Get next run name and increment counter

        Args:
            config_name: optional configuration name to include

        Returns:
            run_name: formatted run name
            run_number: current run number
        """
        run_number = self.counter_data['count'] + 1

        # Format run name
        if config_name:
            run_name = f"run_{run_number}_{config_name}"
        else:
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            run_name = f"run_{run_number}_{timestamp}"

        # Update counter
        self.counter_data['count'] = run_number
        self.counter_data['runs'].append({
            'run_number': run_number,
            'run_name': run_name,
            'config_name': config_name
        })

        self._save_counter()

        return run_name, run_number

    def get_current_count(self):
        """Get current run count"""
        return self.counter_data['count']

    def get_run_history(self):
        """Get history of all runs"""
        return self.counter_data['runs']



