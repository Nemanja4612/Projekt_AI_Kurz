import requests
from dotenv import load_dotenv
import os
from PIL import Image, ImageDraw, ImageFont
import sys
from matplotlib import pyplot as plt
from azure.core.exceptions import HttpResponseError
from azure.ai.vision.imageanalysis import ImageAnalysisClient
from azure.ai.vision.imageanalysis.models import VisualFeatures
from azure.core.credentials import AzureKeyCredential

def main():
    global cv_client

    try:
        # Get Configuration Settings
        load_dotenv()
        ai_endpoint = os.getenv('AI_SERVICE_ENDPOINT')
        ai_key = os.getenv('AI_SERVICE_KEY')

        # Check if an image path is provided
        if len(sys.argv) < 2:
            print("Please provide the path to the image.")
            return

        # Get image path from the command-line argument
        image_file = sys.argv[1]

        # Read the image file
        with open(image_file, "rb") as f:
            image_data = f.read()

        # Authenticate Azure AI Vision client
        cv_client = ImageAnalysisClient(
            endpoint=ai_endpoint,
            credential=AzureKeyCredential(ai_key)
        )

        # Analyze image
        AnalyzeImage(image_file, image_data, cv_client)

        # Background removal
        BackgroundForeground(ai_endpoint, ai_key, image_data)

    except Exception as ex:
        print(ex)

def AnalyzeImage(image_filename, image_data, cv_client):
    print('\nAnalyzing image...')

    try:
        # Get result with specified features to be retrieved
        result = cv_client.analyze(
            image_data=image_data,
            visual_features=[
                VisualFeatures.CAPTION,
                VisualFeatures.DENSE_CAPTIONS,
                VisualFeatures.TAGS,
                VisualFeatures.OBJECTS,
                VisualFeatures.PEOPLE],
        )

    except HttpResponseError as e:
        print(f"Status code: {e.status_code}")
        print(f"Reason: {e.reason}")
        print(f"Message: {e.error.message}")
        return

    # Display analysis results
    if result.caption is not None:
        print("\nCaption:")
        print(" Caption: '{}' (confidence: {:.2f}%)".format(result.caption.text, result.caption.confidence * 100))

    if result.dense_captions is not None:
        print("\nDense Captions:")
        for caption in result.dense_captions.list:
            print(" Caption: '{}' (confidence: {:.2f}%)".format(caption.text, caption.confidence * 100))

    if result.tags is not None:
        print("\nTags:")
        for tag in result.tags.list:
            print(" Tag: '{}' (confidence: {:.2f}%)".format(tag.name, tag.confidence * 100))

    # Load the image and prepare for drawing
    image = Image.open(image_filename)
    draw = ImageDraw.Draw(image)
    color = 'cyan'

    # Annotate detected objects
    if result.objects is not None:
        for detected_object in result.objects.list:
            # Print object name
            print(" {} (confidence: {:.2f}%)".format(detected_object.tags[0].name, detected_object.tags[0].confidence * 100))
     
            # Draw object bounding box
            r = detected_object.bounding_box
            bounding_box = ((r.x, r.y), (r.x + r.width, r.y + r.height))
            draw.rectangle(bounding_box, outline=color, width=3)
            draw.text((r.x, r.y), detected_object.tags[0].name, fill=color)

    outputfile = 'objects.jpg'
    image.save(outputfile)
    print('Results saved in', outputfile)

    # Annotate detected people
    if result.people is not None:
        print("\nPeople in image:")
        num_people = len(result.people.list)
        print(f"Number of people detected: {num_people}")

        # Create a new image for counting people
        image_people_count = Image.open(image_filename)
        draw_people_count = ImageDraw.Draw(image_people_count)
        color = 'red'
        font = ImageFont.load_default()  # You can customize the font if needed

        # Draw rectangles and count number of people
        person_count = 0
        for detected_person in result.people.list:
            # Use bounding box to ensure it's a distinct person detection
            person_confidence = detected_person.confidence  # Ensure this is the correct way to access confidence
            if person_confidence > 0.5:  # Example confidence threshold
                person_count += 1
                r = detected_person.bounding_box
                bounding_box = ((r.x, r.y), (r.x + r.width, r.y + r.height))
                draw_people_count.rectangle(bounding_box, outline=color, width=3)
                draw_people_count.text((r.x, r.y), f'{person_count}', fill=color, font=font)

        # Display the total number of people detected
        draw_people_count.text((20, 20), f'Total People: {person_count}', fill='black', font=font)
        print(f"Filtered number of people detected: {person_count}")

        outputfile_people_count = 'people_count.jpg'
        image_people_count.save(outputfile_people_count)
        print('Results saved in', outputfile_people_count)

def BackgroundForeground(endpoint, key, image_data):
    # Define the API version and mode
    api_version = "2023-02-01-preview"
    mode = "backgroundRemoval"  # Can be "foregroundMatting" or "backgroundRemoval"

    print('\nRemoving background from image...')

    url = f"{endpoint}/computervision/imageanalysis:segment?api-version={api_version}&mode={mode}"

    headers = {
        "Ocp-Apim-Subscription-Key": key,
        "Content-Type": "application/octet-stream"
    }

    response = requests.post(url, headers=headers, data=image_data)

    if response.status_code == 200:
        image = response.content
        with open("backgroundForeground.png", "wb") as file:
            file.write(image)
        print('Results saved in backgroundForeground.png\n')
    else:
        print(f"Error: {response.status_code}")
        print(response.json())

if __name__ == "__main__":
    main()
