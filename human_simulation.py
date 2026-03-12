import numpy as np
import shap

class HumanSimulator:
    """
    Simulates human ethical decision-making using domain rules.
    For each sample requiring intervention, generates multiple options and selects one.
    """
    def __init__(self, rule='fairness_first'):
        self.rule = rule
    
    def generate_options(self, X, original_pred, sensitive):
        """
        Generate 3-5 trade-off options.
        Returns list of (option_value, efficiency_loss, fairness_gain)
        """
        options = []
        # Option 1: Keep original (efficiency-focused)
        options.append((original_pred, 0, 0))
        
        # Option 2: Fairness-adjusted (e.g., increase for unprivileged group)
        adjustment = 0
        if sensitive == 0:  # unprivileged
            adjustment = +5  # arbitrary
        elif sensitive == 1:
            adjustment = -3
        fair_pred = original_pred + adjustment
        options.append((fair_pred, abs(adjustment)/original_pred, 0.1))  # dummy gain
        
        # Option 3: Compromise
        compromise = (original_pred + fair_pred) / 2
        options.append((compromise, abs(compromise-original_pred)/original_pred, 0.05))
        
        # Option 4: Extreme fairness (if needed)
        if sensitive == 0:
            extra_fair = original_pred + 10
            options.append((extra_fair, abs(extra_fair-original_pred)/original_pred, 0.15))
        
        return options
    
    def select_option(self, options, context):
        """
        Simulate human choice. For proof-of-concept, we use a fixed rule.
        """
        if self.rule == 'fairness_first':
            # Choose the option with highest fairness gain (dummy)
            return options[-1][0]  # last option has highest gain
        elif self.rule == 'balanced':
            # Choose compromise
            return options[2][0]
        else:
            # Default: original
            return options[0][0]
    
    def __call__(self, X, pred, sensitive):
        """
        Simulate human decision for a batch of samples.
        Returns adjusted target values.
        """
        adjusted = np.copy(pred)
        for i in range(len(X)):
            opts = self.generate_options(X[i], pred[i], sensitive[i])
            chosen = self.select_option(opts, None)
            adjusted[i] = chosen
        return adjusted