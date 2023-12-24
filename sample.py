import cv2
import numpy as np
import speech_recognition as sr
import tempfile
import platform
import os
from googletrans import Translator
from gtts import gTTS

# Create a temporary directory for audio files
audio_dir = tempfile.mkdtemp()

def translate_text(text, src_lang, dest_lang):
    translator = Translator()
    translation = translator.translate(text, src=src_lang, dest=dest_lang)
    return translation.text

def play_audio(text, dest_lang, src_lang = 'en'):
    print(text)
    try:
        # Translate the text
        translated_text = translate_text(text, src_lang, dest_lang)

        # Create a gTTS object
        tts = gTTS(translated_text, lang=dest_lang)

        # Save the audio to a temporary file within the specified directory
        audio_file = os.path.join(audio_dir, "tmpaudio.mp3")
        tts.save(audio_file)

        # Play the audio using the default system player
        if platform.system() == "Darwin":  # macOS
            os.system(f"open {audio_file}")
        elif platform.system() == "Linux":
            os.system(f"xdg-open {audio_file}")
        elif platform.system() == "Windows":
            os.system(f"start {audio_file}")

    except Exception as e:
        print("Error while playing audio:", str(e))





dic = ('afrikaans', 'af', 'albanian', 'sq',
	'amharic', 'am', 'arabic', 'ar',
	'armenian', 'hy', 'azerbaijani', 'az',
	'basque', 'eu', 'belarusian', 'be',
	'bengali', 'bn', 'bosnian', 'bs', 'bulgarian',
	'bg', 'catalan', 'ca', 'cebuano',
	'ceb', 'chichewa', 'ny', 'chinese (simplified)',
	'zh-cn', 'chinese (traditional)',
	'zh-tw', 'corsican', 'co', 'croatian', 'hr',
	'czech', 'cs', 'danish', 'da', 'dutch',
	'nl', 'english', 'en', 'esperanto', 'eo',
	'estonian', 'et', 'filipino', 'tl', 'finnish',
	'fi', 'french', 'fr', 'frisian', 'fy', 'galician',
	'gl', 'georgian', 'ka', 'german',
	'de', 'greek', 'el', 'gujarati', 'gu',
	'haitian creole', 'ht', 'hausa', 'ha',
	'hawaiian', 'haw', 'hebrew', 'he', 'hindi',
	'hi', 'hmong', 'hmn', 'hungarian',
	'hu', 'icelandic', 'is', 'igbo', 'ig', 'indonesian',
	'id', 'irish', 'ga', 'italian',
	'it', 'japanese', 'ja', 'javanese', 'jw',
	'kannada', 'kn', 'kazakh', 'kk', 'khmer',
	'km', 'korean', 'ko', 'kurdish (kurmanji)',
	'ku', 'kyrgyz', 'ky', 'lao', 'lo',
	'latin', 'la', 'latvian', 'lv', 'lithuanian',
	'lt', 'luxembourgish', 'lb',
	'macedonian', 'mk', 'malagasy', 'mg', 'malay',
	'ms', 'malayalam', 'ml', 'maltese',
	'mt', 'maori', 'mi', 'marathi', 'mr', 'mongolian',
	'mn', 'myanmar (burmese)', 'my',
	'nepali', 'ne', 'norwegian', 'no', 'odia', 'or',
	'pashto', 'ps', 'persian', 'fa',
	'polish', 'pl', 'portuguese', 'pt', 'punjabi',
	'pa', 'romanian', 'ro', 'russian',
	'ru', 'samoan', 'sm', 'scots gaelic', 'gd',
	'serbian', 'sr', 'sesotho', 'st',
	'shona', 'sn', 'sindhi', 'sd', 'sinhala', 'si',
	'slovak', 'sk', 'slovenian', 'sl',
	'somali', 'so', 'spanish', 'es', 'sundanese',
	'su', 'swahili', 'sw', 'swedish',
	'sv', 'tajik', 'tg', 'tamil', 'ta', 'telugu',
	'te', 'thai', 'th', 'turkish',
	'tr', 'ukrainian', 'uk', 'urdu', 'ur', 'uyghur',
	'ug', 'uzbek', 'uz',
	'vietnamese', 'vi', 'welsh', 'cy', 'xhosa', 'xh',
	'yiddish', 'yi', 'yoruba',
	'yo', 'zulu', 'zu')

def takecommand():
	r = sr.Recognizer()
	with sr.Microphone() as source:
		print("Listening...")
		r.pause_threshold = 1
		audio = r.listen(source)

	try:
		print("Recognizing...")
		query = r.recognize_google(audio, language='en-in')
		print(f"The User said {query}\n")
	except Exception as e:

		print("Say that again, please...")
		return "None"
	return query

def destination_language():
	print("Enter the language in which you want to convert: Ex. Hindi, English, etc.")
	print()
	
	# Input destination language
	to_lang = takecommand()
	while (to_lang == "None"):
		to_lang = takecommand()
	to_lang = to_lang.lower()
	return to_lang

#to_lang = destination_language()

# Mapping it with the code
''''
while (to_lang not in dic):
	print("Language in which you are trying to convert is currently not available, please input some other language")
	print()
	to_lang = destination_language()
to_lang = dic[dic.index(to_lang) + 1]

'''

to_lang = input('Preferred language :').lower()
to_lang = dic[dic.index(to_lang) + 1]
# Assuming you have rough estimations
# Real height of the object (in meters)
real_object_height = 0.2  # Example: 20 cm

# Focal length of the camera (in pixels)
focal_length = 1000

# Initialize a dictionary to keep track of voiced objects and their last detection status
voiced_objects = {}

# Load YOLOv3 weights and configuration file
cfg = r"C:\Users\JAYAVELU A\Desktop\IFP\yolov3.cfg"
weight = r"C:\Users\JAYAVELU A\Desktop\IFP\yolov3.weights"
net = cv2.dnn.readNetFromDarknet(cfg, weight)

# Load list of classes
classes = []
with open(r"C:\Users\JAYAVELU A\Desktop\IFP\coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Set the minimum confidence threshold for detecting objects
conf_threshold = 0.5

# Set the non-maximum suppression threshold for eliminating overlapping detections
nms_threshold = 0.4

# Open camera
cap = cv2.VideoCapture(0)



while True:
    # Read frame from the camera
    ret, frame = cap.read()

    # Create a blob from the input image
    blob = cv2.dnn.blobFromImage(frame, 1/255, (416, 416), (0, 0, 0), True, crop=False)

    # Set the input for the neural network
    net.setInput(blob)

    # Get the output layer names
    output_layers = net.getUnconnectedOutLayersNames()

    # Run forward pass through the network
    outs = net.forward(output_layers)

    # Initialize lists for detected objects' class IDs, confidences, and bounding boxes
    class_ids = []
    confidences = []
    boxes = []

    # Loop over all detected objects
    for out in outs:
        for detection in out:
            # Get class ID and confidence
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            # Filter out weak detections
            if confidence > conf_threshold:
                # Get the bounding box coordinates
                box = detection[0:4] * np.array([frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]])
                (centerX, centerY, width, height) = box.astype("int")

                # Add the detected object's class ID, confidence, and bounding box to lists
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))
                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([x, y, int(width), int(height)])

    # Apply non-maximum suppression to eliminate overlapping detections
    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

    # Draw the bounding boxes and labels for each detected object
    for i in indices:
        box = boxes[i]
        x = box[0]
        y = box[1]
        w = box[2]
        h = box[3]
        label = f"{classes[class_ids[i]]}: {confidences[i]:.2f}"
        color = (0, 255, 0)
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(frame, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        object_name = classes[class_ids[i]]

        # Check if the object was previously voiced or not detected
        if object_name not in voiced_objects or not voiced_objects[object_name]:
            voiced_objects[object_name] = True

            # Calculate estimated distance
            estimated_distance = (real_object_height * focal_length) / h  # Using simple triangle similarity
            print(object_name, "Distance:", estimated_distance, "meters")

            # Generate and play the audio
            text_to_speak = f'{object_name} is approximately {estimated_distance:.2f} meters away.'
            play_audio(text_to_speak, to_lang)

    # Update voiced status for objects that are not detected
    for object_name in voiced_objects.keys():
        if object_name not in [classes[class_ids[i]] for i in indices]:
            voiced_objects[object_name] = False

    # Display the resulting image
    cv2.imshow("object detection", frame)

    # Break the loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
