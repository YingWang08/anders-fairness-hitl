import numpy as np

class HumanSimulator:
    """Rule-based simulated human decision-maker for ethical interventions."""
    def __init__(self, fairness_threshold=0.8, max_efficiency_loss=0.1):
        self.fairness_threshold = fairness_threshold
        self.max_efficiency_loss = max_efficiency_loss

    def decide(self, options, context):
        """
        options: list of dicts with keys 'value' (prediction), 'di', 'efficiency_loss', 'description'
        context: dict with keys like 'region', 'avg_allocation_B', etc.
        Returns: selected option index, decision_basis string
        """
        # Rule 1: Reject options that explicitly discriminate against protected group (simplified)
        # For agricultural scenario, if region is A and allocation is too low compared to B average
        valid_options = []
        for opt in options:
            reject = False
            if context.get('region') == 'A':
                # If option allocates less than 70% of average B allocation, reject
                if 'avg_allocation_B' in context and opt['value'] < context['avg_allocation_B'] * 0.7:
                    reject = True
            if not reject:
                valid_options.append(opt)

        if not valid_options:
            # Fallback: first option
            return 0, "fallback (no valid options)"

        # Rule 2: Prefer options that restore DI to >=0.85 with efficiency loss <=10%
        best_idx = None
        best_score = -np.inf
        for i, opt in enumerate(valid_options):
            if opt['di'] >= 0.85 and opt['efficiency_loss'] <= 0.1:
                # Among those, choose smallest efficiency loss
                score = -opt['efficiency_loss']
                if score > best_score:
                    best_score = score
                    best_idx = i

        if best_idx is not None:
            return best_idx, "fairness+lowcost"

        # Rule 3: Otherwise choose option with highest DI
        best_idx = np.argmax([opt['di'] for opt in valid_options])
        return best_idx, "max_fairness"