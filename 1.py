import os
import torch
import torchvision.transforms as transforms
from transformers import AutoModelForImageClassification
from PIL import Image
import warnings

# Suppress specific warnings that might clutter output
warnings.filterwarnings("ignore", category=UserWarning, module='torchvision.transforms.functional_tensor')


def install_and_check_dependencies():
    """
    Checks if necessary libraries are installed and installs them if not.
    """
    print("Checking and installing dependencies...")
    required_packages = ['torch', 'torchvision', 'Pillow', 'transformers']
    for package in required_packages:
        try:
            __import__(package)
            print(f"{package} is already installed.")
        except ImportError:
            print(f"{package} not found. Installing...")
            try:
                os.system(f"pip install {package}")
                print(f"{package} installed successfully.")
            except Exception as e:
                print(f"Error installing {package}: {e}")
                print("Please install it manually: pip install {package}")
                exit()
    print("All dependencies checked.")


def get_food101_labels():
    """
    Returns the official list of 101 food categories from the Food101 dataset.
    This list MUST be in the correct order as the model was trained on.
    """
    food101_labels = [
        'apple_pie', 'baby_back_ribs', 'baklava', 'beef_carpaccio', 'beef_tartare',
        'beet_salad', 'beignets', 'bibimbap', 'bread_pudding', 'breakfast_burrito',
        'broccoli_cheese_soup', 'bruschetta', 'caesar_salad', 'cannoli', 'caprese_salad',
        'carrot_cake', 'ceviche', 'cheesecake', 'cheese_plate', 'chicken_curry',
        'chicken_quesadilla', 'chicken_wings', 'chocolate_cake', 'chocolate_mousse',
        'churros', 'clam_chowder', 'club_sandwich', 'crab_cakes', 'creme_brulee',
        'croque_madame', 'cup_cakes', 'deviled_eggs', 'donuts', 'dumplings',
        'edamame', 'eggs_benedict', 'escargots', 'falafel', 'filet_mignon',
        'fish_and_chips', 'foie_gras', 'french_fries', 'french_onion_soup', 'french_toast',
        'fried_calamari', 'fried_rice', 'frozen_yogurt', 'garlic_bread', 'gnocchi',
        'greek_salad', 'grilled_cheese_sandwich', 'grilled_salmon', 'guacamole',
        'hamburger', 'hot_dog', 'ice_cream', 'lasagna', 'lobster_bisque', 'lobster_roll_sandwich',
        'macaroni_and_cheese', 'macarons', 'miso_soup', 'mussels', 'nachos',
        'omelette', 'onion_rings', 'oysters', 'pad_thai', 'paella', 'pancakes',
        'panna_cotta', 'peking_duck', 'pho', 'pizza', 'pork_chop', 'poutine',
        'prime_rib', 'pulled_pork_sandwich', 'ramen', 'ravioli', 'red_velvet_cake',
        'risotto', 'samosa', 'sashimi', 'scallops', 'seaweed_salad', 'shrimp_scampi',
        'smores', 'spaghetti_bolognese', 'spaghetti_carbonara', 'spring_rolls',
        'steak', 'strawberry_shortcake', 'sushi', 'tacos', 'takoyaki', 'tiramisu',
        'tuna_tartare', 'waffles'
    ]
    return food101_labels


def recognize_food_food101(image_path="meal.jpg"):
    """
    Recognizes food in the given image using an EfficientNet-B1 model
    fine-tuned on the Food101 dataset. Does NOT estimate quantity.

    Args:
        image_path (str): Path to the input image file (e.g., 'meal.jpg').
    """
    if not os.path.exists(image_path):
        print(f"Error: Image file '{image_path}' not found in the current directory.")
        return

    # 1. Load fine-tuned EfficientNet-B1 model
    model_name = "gabrielganan/efficientnet_b1-food101"
    print(f"Loading fine-tuned {model_name} model (this may take a moment)...")

    try:
        model = AutoModelForImageClassification.from_pretrained(model_name)
        model.eval()  # Set model to evaluation mode
    except Exception as e:
        print(f"Error loading model from Hugging Face: {e}")
        print("Please ensure you have an active internet connection to download the model.")
        return

    # Determine if CUDA is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"Using device: {device}")

    # 2. Manually define preprocessing transformations for EfficientNet-B1
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(240),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    # 3. Load the image
    try:
        img = Image.open(image_path).convert("RGB")
        print(f"Image '{image_path}' loaded successfully.")
    except Exception as e:
        print(f"Error loading image '{image_path}': {e}")
        return

    # 4. Preprocess and move to device
    input_tensor = preprocess(img).unsqueeze(0).to(device)

    # 5. Perform inference
    print("Performing food recognition...")
    with torch.no_grad():
        output = model(input_tensor).logits

    # 6. Get predicted class using our explicit Food101 labels
    probabilities = torch.nn.functional.softmax(output, dim=1)

    # Get the official Food101 labels
    food101_labels = get_food101_labels()

    top5_prob, top5_indices = torch.topk(probabilities, 5)

    print(f"\n--- Food Recognition Results ({model_name} - Food101 Fine-tuned) ---")
    print("This model is specifically trained to classify 101 Food101 categories.")

    for i in range(top5_prob.size(1)):
        predicted_idx = top5_indices[0][i].item()
        predicted_prob = top5_prob[0][i].item()

        # Use our manually defined Food101 labels
        if predicted_idx < len(food101_labels):
            predicted_label = food101_labels[predicted_idx]
        else:
            predicted_label = f"Unknown Label Index {predicted_idx}"  # Fallback if index out of bounds

        print(f"Top {i + 1} prediction: {predicted_label} (Confidence: {predicted_prob:.2f})")

    print("\n--- Quantity Estimation (Conceptual) ---")
    print("Accurate quantity estimation from a single 2D image is extremely challenging.")
    print("This script provides food *recognition* (classification) only.")
    print("To estimate quantity, advanced techniques are required, such as:")
    print("  - Depth estimation (3D reconstruction)")
    print("  - Instance segmentation (precise outlines of each food item)")
    print("  - Using a known reference object in the image (e.g., a standard-sized plate or coin)")
    print("  - Specialized models trained on datasets with quantity annotations.")
    print("For a complex dish like 'mushroom risotto', estimating quantity would involve:")
    print("  1. Identifying the 'risotto' portion accurately.")
    ("  2. Estimating its volume (e.g., by inferring depth and dimensions).")
    ("  3. Converting volume to weight using a density database for risotto.")
    print("This level of analysis is beyond the scope of a simple classification model.")


if __name__ == "__main__":
    install_and_check_dependencies()
    recognize_food_food101("meal.jpg")