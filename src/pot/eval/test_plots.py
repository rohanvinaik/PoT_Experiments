import matplotlib.pyplot as plt
import numpy as np

from pot.eval.plots import plot_det_curve

# Add parent directory to path for pot imports
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))



def test_plot_det_curve_labels_and_save(tmp_path):
    far = np.linspace(0.01, 0.99, 10)
    frr = 1 - far
    save_file = tmp_path / "det_curve.png"

    fig = plot_det_curve(far, frr, title="Test DET", save_path=str(save_file))

    assert isinstance(fig, plt.Figure)
    ax = fig.axes[0]
    assert ax.get_xlabel() == "False Accept Rate (FAR)"
    assert ax.get_ylabel() == "False Reject Rate (FRR)"
    assert save_file.exists()

    plt.close(fig)

