import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import Dict, List

def plot_roc_curve(far_values: List[float], frr_values: List[float], title: str = "ROC Curve"):
    """Plot ROC curve"""
    plt.figure(figsize=(8, 6))
    plt.plot(far_values, 1 - np.array(frr_values), 'b-', linewidth=2)
    plt.xlabel('False Accept Rate')
    plt.ylabel('True Accept Rate')
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.show()

def plot_det_curve(far_values: List[float], frr_values: List[float], title: str = "DET Curve"):
    """Plot Detection Error Tradeoff curve"""
    # TODO: implement DET curve with probit scale
    pass

def plot_auroc_vs_queries(query_budgets: List[int], auroc_values: List[float]):
    """Plot AUROC vs query budget"""
    plt.figure(figsize=(8, 6))
    plt.plot(query_budgets, auroc_values, 'o-', linewidth=2)
    plt.xlabel('Query Budget')
    plt.ylabel('AUROC')
    plt.title('AUROC vs Query Budget')
    plt.grid(True, alpha=0.3)
    plt.show()

def plot_leakage_curve(rho_values: List[float], detection_rates: List[float]):
    """Plot detection rate vs leakage fraction"""
    plt.figure(figsize=(8, 6))
    plt.plot(rho_values, detection_rates, 'r^-', linewidth=2)
    plt.xlabel('Leakage Fraction (ρ)')
    plt.ylabel('Detection Rate')
    plt.title('Detection vs Leakage')
    plt.grid(True, alpha=0.3)
    plt.show()