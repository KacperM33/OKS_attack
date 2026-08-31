import numpy as np


def forward_perturbation(epsilon, current_adversarial, original_image):
    """
    Performs a 'forward step' towards the target sample.

    Args:
        epsilon: Step size (scaling factor).
        current_adversarial: The current sample (e.g., adversarial_sample).
        original_image: The target sample towards which the step is taken (e.g., the original image).
    """
    diff = (original_image - current_adversarial).astype(np.float32)
    distance = get_diff(original_image, current_adversarial)
    perturb = (diff / distance) * epsilon
    return perturb


def orthogonal_perturbation(delta, current_adversarial, original_image):
    """
    Generates an orthogonal perturbation relative to the direction (target - prev).

    Args:
        delta: Scale factor (maximum length of the perturbation).
        current_adversarial: The current image (e.g., adversarial_sample).
        original_image: The target image (e.g., the original image).
    """
    current_adv = current_adversarial.astype(np.float32)
    original_img = original_image.astype(np.float32)

    diff = original_img - current_adv
    distance = get_diff(original_img, current_adv)

    diff_normalized = diff / distance

    perturb = np.random.randn(*current_adv.shape).astype(np.float32)
    perturb /= get_diff(perturb, np.zeros_like(perturb))

    dot_product = np.sum(perturb * diff_normalized)
    perturb -= dot_product * diff_normalized

    perturb /= get_diff(perturb, np.zeros_like(perturb))

    perturb *= (delta * distance)

    return perturb


def get_diff(sample_1, sample_2):
    diff = sample_1.astype(np.float32) - sample_2.astype(np.float32)
    return np.sqrt(np.sum(np.square(diff))) + 1e-9