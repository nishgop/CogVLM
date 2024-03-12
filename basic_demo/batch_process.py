import argparse
import csv
import os
from pathlib import Path
from cli_demo_sat import initialize_model, chat 

def batch_process_images(directory, model, image_processor, text_processor, args):
    responses = []
    for image_path in Path(directory).rglob('*.jpg'):  # Adjust pattern as necessary
        # Adapt the function call based on your actual implementation
        # Here, assume chat function or similar is adapted to be non-interactive and returns structured data (e.g., JSON)
        response = chat(image_path=image_path, model=model, text_processor=text_processor, img_processor=image_processor, args=args)
        responses.append({"image_path": str(image_path), "response": response})
    return responses

def save_responses_to_csv(responses, csv_file_path):
    with open(csv_file_path, mode='w', newline='') as csv_file:
        fieldnames = ['image_path', 'response']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for response in responses:
            writer.writerow(response)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Batch process images and save responses to a CSV file.')
    parser.add_argument('--image_directory', type=str, required=True, help='Directory containing the images to process.')
    parser.add_argument('--csv_output_path', type=str, required=True, help='Path to save the output CSV file.')
    # Add additional arguments as necessary for model initialization

    args = parser.parse_args()

    # Initialize the model using the provided arguments
    model, image_processor, text_processor = initialize_model(args)

    # Now use the provided command line arguments for directory paths
    responses = batch_process_images(args.image_directory, model, image_processor, text_processor, args)
    save_responses_to_csv(responses, args.csv_output_path)


