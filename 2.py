import os
import torch
from transformers import AutoImageProcessor, AutoModelForImageClassification
from PIL import Image
import torchvision.transforms as transforms
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


def get_predictions(model_name, image_path, device, autocast_dtype, use_auto_processor=True):
    """
    Loads a model and its processor, performs inference, and returns probabilities.
    Handles different preprocessing based on model type.
    """
    print(f"\n--- Loading and inferring with {model_name} ---")

    try:
        model = AutoModelForImageClassification.from_pretrained(model_name)
        model.eval().to(device)

        img = Image.open(image_path).convert("RGB")

        if use_auto_processor:
            processor = AutoImageProcessor.from_pretrained(model_name)
            inputs = processor(img, return_tensors="pt").to(device)
        else:  # For EfficientNet-B1, which had issues with AutoImageProcessor previously
            # Manual transforms for EfficientNet-B1 (240x240 input)
            transform = transforms.Compose([
                transforms.Resize((240, 240)),  # EfficientNet-B1 expects 240x240
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
            inputs = transform(img).unsqueeze(0).to(device)  # Add batch dimension

        with torch.no_grad():
            with torch.autocast(device_type=device.type, dtype=autocast_dtype):
                output = model(**inputs).logits

        probabilities = torch.nn.functional.softmax(output, dim=1)
        return probabilities

    except Exception as e:
        print(f"Error during inference with {model_name}: {e}")
        print("This model's predictions will be skipped from the ensemble.")
        return None
    finally:
        # Explicitly delete model to free VRAM for the next one
        if 'model' in locals():
            del model
            if device.type == 'cuda':
                torch.cuda.empty_cache()


def recognize_food_ensemble_with_topk(image_path="meal.jpg"):
    """
    Recognizes food in the given image using an ensemble of two Food101-fine-tuned models:
    1. gabrielganan/efficientnet_b1-food101
    2. Mullerjo/food-101-finetuned-model (ViT-Base)
    Predictions are combined via averaging probabilities and top-K results are shown.
    Does NOT estimate quantity.

    Args:
        image_path (str): Path to the input image file (e.g., 'meal.jpg').
    """
    if not os.path.exists(image_path):
        print(f"Error: Image file '{image_path}' not found in the current directory.")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    autocast_dtype = torch.float16 if device.type == 'cuda' else torch.float32
    if device.type == 'cuda':
        print("Enabling mixed precision (FP16) for inference where possible.")

    all_probabilities = []

    # Model 1: EfficientNet-B1
    model1_name = "gabrielganan/efficientnet_b1-food101"
    probs1 = get_predictions(model1_name, image_path, device, autocast_dtype, use_auto_processor=False)
    if probs1 is not None:
        all_probabilities.append(probs1)

    # Model 2: ViT-Base (Mullerjo)
    model2_name = "Mullerjo/food-101-finetuned-model"
    probs2 = get_predictions(model2_name, image_path, device, autocast_dtype, use_auto_processor=True)
    if probs2 is not None:
        all_probabilities.append(probs2)

    if not all_probabilities:
        print("No models successfully made predictions. Cannot ensemble.")
        return

    # Ensemble the predictions by averaging probabilities
    print("\n--- Averaging predictions from all successful models ---")
    if len(all_probabilities) == 1:
        print("Only one model successfully made predictions. No true ensembling.")
        ensembled_probabilities = all_probabilities[0]
    else:
        # Sum all probability tensors and then divide by the number of models
        ensembled_probabilities = torch.sum(torch.stack(all_probabilities), dim=0) / len(all_probabilities)

    # Get the official Food101 labels
    food101_labels = get_food101_labels()

    # Get top 5 predictions for the ensemble result
    top5_prob, top5_indices = torch.topk(ensembled_probabilities, 5)

    print(f"\n--- Final Ensemble Food Recognition Results (Food101 Fine-tuned Models) ---")
    print(f"Ensembled from {len(all_probabilities)} model(s).")
    print("This ensemble is specifically trained to classify 101 Food101 categories.")
    print("\nTop 5 Predictions:")

    for i in range(top5_prob.size(1)):
        predicted_idx = top5_indices[0][i].item()
        predicted_prob = top5_prob[0][i].item()

        if predicted_idx < len(food101_labels):
            predicted_label = food101_labels[predicted_idx]
        else:
            predicted_label = f"Unknown Label Index {predicted_idx}"

        print(f"  {i + 1}. {predicted_label} (Confidence: {predicted_prob:.2f})")

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
    print("  2. Estimating its volume (e.g., by inferring depth and dimensions).")
    print("  3. Converting volume to weight using a density database for risotto.")
    print("This level of analysis is beyond the scope of a simple classification model.")


if __name__ == "__main__":
    install_and_check_dependencies()
    recognize_food_ensemble_with_topk("meal.jpg")