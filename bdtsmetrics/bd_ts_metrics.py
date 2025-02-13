import argparse
import yaml
import os
import sys
import json
from jinja2 import Environment, FileSystemLoader
import base64
from git import Repo
from .src.preprocess import load_my_own_data_csv, load_my_own_data_numpy, load_my_own_data_csv_v2
from .src.evaluation import evaluate_data
from .src.utils import write_json_data

class tsMetrics:
    """Implements time series metrics in python
      Usage:
      1. evaluate the quality of synthetic time series data

      :param config: metric config file.
      :type config: str
      :param real_data:  real time series data file
      :type real_data: str
      :param syn_data: synthetic time series data file
      :type syn_data: str
      :param seq_len: sequence length
      :type seq_len: int
    """

    def __init__(self, config, real_data, syn_data, seq_len):
      self.config = config
      self.real_data = real_data
      self.syn_data = syn_data
      self.seq_len = seq_len


    def load_config_from_file(self,config_file):
        with open(config_file, 'r') as file:
            return yaml.safe_load(file)


    def evaluate(self):
        config = self.load_config_from_file(self.config)
        if self.real_data.endswith(".csv"):
            data = load_my_own_data_csv(self.real_data, self.seq_len)
        elif self.real_data.endswith(".npy"):
            data = load_my_own_data_numpy(self.real_data)
        else:
            raise Exception("Real data: wrong data format")

        if self.syn_data.endswith(".csv"):
            generated_data = load_my_own_data_csv(self.syn_data, self.seq_len)
        elif self.syn_data.endswith(".npy"):
            generated_data = load_my_own_data_numpy(self.syn_data)
        else:
            raise Exception("Synthetic data: wrong data format")


        results = evaluate_data(config['evaluation'], data, generated_data)

        if not os.path.isdir(os.path.join(os.getcwd(), 'result')):
            os.mkdir(os.path.join(os.getcwd(), 'result'))

        with open('./result/result.json', 'w') as f:
            json.dump(results, f)

        print('Program normal end.')

        # Code below generates the report (.html file) based on the results stored in the result folder
        # Define the root folder (where main.py and template.html are located)
        root_folder = os.path.abspath(".")

        # Define the result folder (subfolder where result.json and PNG images reside)
        result_folder = os.path.join(root_folder, "result")

        # Load the JSON file from the result folder (assumed to be named "result.json")
        json_path = os.path.join(result_folder, "result.json")
        with open(json_path, "r", encoding="utf-8") as f:
            data_metrics = json.load(f)

        # List all PNG files in the result folder and encode them as Base64 strings
        png_files = sorted([f for f in os.listdir(result_folder) if f.lower().endswith('.png')])
        images = []
        for f in png_files:
            file_path = os.path.join(result_folder, f)
            with open(file_path, "rb") as img_file:
                # Read the binary data and encode it as Base64
                encoded_string = base64.b64encode(img_file.read()).decode("utf-8")
                # Create a data URL for embedding the image directly in the HTML
                data_url = f"data:image/png;base64,{encoded_string}"
                images.append({
                    "src": data_url,
                    "caption": f  # Using the file name as the caption; modify if needed.
                })

        repo_url = "https://github.com/bxz9200/bd-ts-metrics.git"  # Replace with your repo URL
        local_repo_path = "temp_templates"  # A temporary directory to clone into

        try:
            # 1. Clone the repository (if it doesn't exist locally):
            if not os.path.exists(local_repo_path):
                Repo.clone_from(repo_url, local_repo_path)
            else:
                repo = Repo(local_repo_path)
                repo.remotes.origin.pull()  # Update the repo if it exists

            # 2. Set up the Jinja2 environment:
            env = Environment(loader=FileSystemLoader(local_repo_path))

            # 3. Load the template (relative to the cloned directory):
            template = env.get_template("template.html")  # Path relative to repo root

            # 4. Render the template with data from JSON, the list of images, and the entire JSON as json_data.
            rendered_html = template.render(
                title=data_metrics.get("title", "Betterdata TimeSeries Report"),
                heading=data_metrics.get("heading", "Welcome"),
                images=images,
                json_data=data_metrics
            )

            # Write the rendered HTML to an output file in the root folder
            output_path = os.path.join(root_folder, "Report.html")
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(rendered_html)

            print(f"HTML file generated: {output_path}")


        finally:
            pass
        # 5. (Optional) Remove the temporary directory after you're done:
        # shutil.rmtree(local_repo_path)  # Be careful with this, only if you're sure you want to delete it.






