from audiomentations import Compose, AddGaussianNoise, TimeMask, TanhDistortion, ClippingDistortion,BitCrush, PeakingFilter, PitchShift, Shift,LowPassFilter, HighPassFilter,Limiter

def get_augmentations():
    """
    Returns a Compose object containing the desired augmentations.
    """
    augment = Compose([
        AddGaussianNoise(p=0.99,max_amplitude=0.25),
        TimeMask(p=0.25),
        # Shift(p=0.5)
    ])
    return augment
