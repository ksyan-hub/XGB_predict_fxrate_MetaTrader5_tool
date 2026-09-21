from datetime import datetime

import os

class cmd_config:
    def __init__(self):

        self.exit_run = "exit_run"

class build_file_path:
    def __init__(self, start, end):
        self.start_str = datetime(*start).strftime("%Y%m%d")
        self.end_str = datetime(*end).strftime("%Y%m%d")

    def train_data(self):
        return rf".\csv_rate_data\train_data_rates\use_train_data_from{self.start_str}_to{self.end_str}.csv"

    def ran_data(self):
        return rf".\csv_rate_data\running_data_rates\use_train_data_from{self.start_str}_to{self.end_str}.csv"               

    def predictions_folder(self):
        return rf".\prediction_rate_data\trained_model\trained_model_predictions_{self.start_str}_{self.end_str}"

    def trained_model_predictions_high(self):
        os.makedirs(rf".\prediction_rate_data\trained_model\trained_model_predictions_{self.start_str}_{self.end_str}", exist_ok = True)
        return rf".\prediction_rate_data\trained_model\trained_model_predictions_{self.start_str}_{self.end_str}\trained_model_predictions_high_{self.start_str}_{self.end_str}.csv"
 
    def trained_model_predictions_low(self):
        os.makedirs(rf".\prediction_rate_data\trained_model\trained_model_predictions_{self.start_str}_{self.end_str}", exist_ok = True)
        return rf".\prediction_rate_data\trained_model\trained_model_predictions_{self.start_str}_{self.end_str}\trained_model_predictions_low_{self.start_str}_{self.end_str}.csv"

    def save_folder(self):
        return rf".\saved_model\trained_model\saved_trained_model_{self.start_str}_{self.end_str}"

    def save_trained_model_high(self):
        os.makedirs(rf".\saved_model\trained_model\saved_trained_model_{self.start_str}_{self.end_str}", exist_ok = True)
        return rf".\saved_model\trained_model\saved_trained_model_{self.start_str}_{self.end_str}\saved_train_model_high_{self.start_str}_{self.end_str}.json"

    def save_trained_model_low(self):
        os.makedirs(rf".\saved_model\trained_model\saved_trained_model_{self.start_str}_{self.end_str}", exist_ok = True)
        return rf".\saved_model\trained_model\saved_trained_model_{self.start_str}_{self.end_str}\saved_train_model_low_{self.start_str}_{self.end_str}.json"

    def save_ran_model_high(self):
        os.makedirs(rf".\saved_model\ran_model\saved_trained_model_{self.start_str}_{self.end_str}", exist_ok = True)
        return rf".\saved_model\ran_model\saved_trained_model_{self.start_str}_{self.end_str}\ran_model_high_{self.start_str}_{self.end_str}.json"

    def save_ran_model_low(self):
        os.makedirs(rf".\saved_model\ran_model\saved_trained_model_{self.start_str}_{self.end_str}", exist_ok = True)
        return rf".\saved_model\ran_model\saved_trained_model_{self.start_str}_{self.end_str}\ran_model_low_{self.start_str}_{self.end_str}.json"