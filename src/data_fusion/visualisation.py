import cv2
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize


def display_heatmap_with_colorbar(fused_heatmap, z_min, z_max):
    """Display heatmap with colorbar showing height values with JET colormap (blue to red).
    Args:
        fused_heatmap (np.ndarray): The heatmap image to display. Shape: (H, W, 3).
        z_min (float): Minimum height value for colorbar.
        z_max (float): Maximum height value for colorbar.
    
    """
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Display the RGB image
    ax.imshow(cv2.cvtColor(fused_heatmap, cv2.COLOR_BGR2RGB))
    ax.set_title("Height Map Overlay on Image (Blue=Low, Red=High)", fontsize=14, fontweight="bold")
    ax.axis("off")
    
    # Create ScalarMappable with JET colormap to match the heatmap colors
    sm = ScalarMappable(cmap=plt.cm.jet, norm=Normalize(vmin=z_min, vmax=z_max))
    sm.set_array([])
    
    # Create colorbar with the JET colormap
    cbar = plt.colorbar(sm, ax=ax, orientation="vertical", pad=0.02, fraction=0.046)
    cbar.set_label(f"Height (units)\n[{z_min:.2f} to {z_max:.2f}]", fontsize=12, fontweight="bold")
    
    # Set colorbar tick labels to represent actual height values
    cbar_ticks = np.linspace(z_min, z_max, 5)  # 5 ticks with actual height values
    cbar.set_ticks(cbar_ticks)
    cbar.ax.set_yticklabels([f"{h:.2f}" for h in cbar_ticks])
    
    plt.tight_layout()
    plt.show()