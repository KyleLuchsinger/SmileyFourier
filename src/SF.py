import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.patches as patches
from matplotlib import patheffects
from scipy import signal, interpolate
import tqdm
import time


def generate_smiley_outline():
    print("Generating smiley face outline...")
    # Parameter for outline point density
    n_points = 3000

    # Generate main circle outline (face)
    theta_face = np.linspace(0, 2 * np.pi, int(n_points * 0.4), endpoint=False)
    radius_face = 1.0
    x_face = radius_face * np.cos(theta_face)
    y_face = radius_face * np.sin(theta_face)

    # Generate curved smile with radius modulation for improved appearance
    theta_smile = np.linspace(0.2 * np.pi, 0.8 * np.pi, int(n_points * 0.2), endpoint=False)
    radius_smile = 0.6 + 0.05 * np.sin(theta_smile * 2)
    x_smile = radius_smile * np.cos(theta_smile)
    y_smile = radius_smile * np.sin(theta_smile) - 0.2
    # Invert y-coordinates for correct orientation
    y_smile = -y_smile

    # Generate smile edge dimples for enhanced detail
    left_dimple_x = np.linspace(-0.6, -0.5, int(n_points * 0.05))
    left_dimple_y = -0.5 * np.sin(np.linspace(0, np.pi, int(n_points * 0.05))) - 0.1

    right_dimple_x = np.linspace(0.5, 0.6, int(n_points * 0.05))
    right_dimple_y = -0.5 * np.sin(np.linspace(0, np.pi, int(n_points * 0.05))) - 0.1

    # Generate left eye outline
    theta_left_eye = np.linspace(0, 2 * np.pi, int(n_points * 0.1), endpoint=False)
    radius_left_eye = 0.15
    x_left_eye = radius_left_eye * np.cos(theta_left_eye) - 0.4
    y_left_eye = radius_left_eye * np.sin(theta_left_eye) + 0.4

    # Generate left pupil detail
    theta_left_pupil = np.linspace(0, 2 * np.pi, int(n_points * 0.05), endpoint=False)
    radius_left_pupil = 0.05
    x_left_pupil = radius_left_pupil * np.cos(theta_left_pupil) - 0.4
    y_left_pupil = radius_left_pupil * np.sin(theta_left_pupil) + 0.4

    # Generate right eye outline
    theta_right_eye = np.linspace(0, 2 * np.pi, int(n_points * 0.1), endpoint=False)
    radius_right_eye = 0.15
    x_right_eye = radius_right_eye * np.cos(theta_right_eye) + 0.4
    y_right_eye = radius_right_eye * np.sin(theta_right_eye) + 0.4

    # Generate right pupil detail
    theta_right_pupil = np.linspace(0, 2 * np.pi, int(n_points * 0.05), endpoint=False)
    radius_right_pupil = 0.05
    x_right_pupil = radius_right_pupil * np.cos(theta_right_pupil) + 0.4
    y_right_pupil = radius_right_pupil * np.sin(theta_right_pupil) + 0.4

    # Generate curved eyebrows
    left_eyebrow_x = np.linspace(-0.5, -0.3, int(n_points * 0.05))
    left_eyebrow_y = 0.1 * np.sin(np.linspace(0, np.pi, int(n_points * 0.05))) + 0.55

    right_eyebrow_x = np.linspace(0.3, 0.5, int(n_points * 0.05))
    right_eyebrow_y = 0.1 * np.sin(np.linspace(0, np.pi, int(n_points * 0.05))) + 0.55

    # Combine all coordinate sets
    x = np.concatenate([
        x_face, x_smile, x_left_eye, x_right_eye,
        x_left_pupil, x_right_pupil, left_eyebrow_x, right_eyebrow_x,
        left_dimple_x, right_dimple_x
    ])
    y = np.concatenate([
        y_face, y_smile, y_left_eye, y_right_eye,
        y_left_pupil, y_right_pupil, left_eyebrow_y, right_eyebrow_y,
        left_dimple_y, right_dimple_y
    ])

    print("✅ Smiley face outline generated!")
    return x, y


def compute_fourier_coefficients(x, y, num_coeffs=200):
    """Compute Fourier coefficients for a curve defined by x and y coordinates."""
    print(f"Computing {num_coeffs * 2 + 1} Fourier coefficients...")
    start_time = time.time()

    # Convert to complex representation
    z = x + 1j * y
    n = len(z)

    # Parameter space
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)

    coeffs = []
    # Utilize tqdm for visualization of computation progress
    for k in tqdm.tqdm(range(-num_coeffs, num_coeffs + 1), desc="Calculating coefficients"):
        c_k = np.sum(z * np.exp(-1j * k * t)) / n
        coeffs.append((k, c_k))

    # Sort by coefficient magnitude for optimization of visual representation
    sorted_coeffs = sorted(coeffs, key=lambda c: abs(c[1]), reverse=True)

    duration = time.time() - start_time
    print(f"✅ Fourier coefficients computed in {duration:.2f} seconds!")
    return sorted_coeffs


def animate_fourier_drawing(coeffs, num_terms=150, duration_sec=15, fps=20):
    """Create an animation of the Fourier drawing process."""
    print(f"Setting up animation with {num_terms} terms, {duration_sec}s duration at {fps} fps...")

    # Initialize figure parameters
    fig, ax = plt.subplots(figsize=(10, 10), facecolor='white')
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_aspect('equal')
    ax.axis('off')  # Remove axes for clean display

    # Initialize epicycle visualization elements
    circles = []
    lines = []

    # Initialize traced curve representation
    path_line, = ax.plot([], [], 'black', lw=3.0, path_effects=[
        patheffects.SimpleLineShadow(shadow_color='gray', alpha=0.3),
        patheffects.Normal()
    ])
    traced_x, traced_y = [], []

    # Calculate total frame count
    num_frames = duration_sec * fps

    # Initialize progress tracking elements
    progress_text = ax.text(0.02, 0.02, '', transform=ax.transAxes,
                            fontsize=12, color='blue', bbox=dict(facecolor='white', alpha=0.7))
    frames_completed = 0
    start_time = None

    def init():
        # Initialize visual elements for epicycles and connecting lines
        for _ in range(num_terms):
            circle = patches.Circle((0, 0), 0.1, fill=False, color='blue', alpha=0.3)
            line, = ax.plot([], [], 'red', lw=1, alpha=0.5)
            ax.add_patch(circle)
            circles.append(circle)
            lines.append(line)
        progress_text.set_text('0%')
        return [path_line, progress_text] + circles + lines

    def update(frame):
        nonlocal frames_completed, start_time

        # Initialize timer on first frame
        if start_time is None:
            start_time = time.time()

        # Calculate parametric time value for current frame
        t = 2 * np.pi * frame / num_frames

        # Initialize center point
        x, y = 0, 0

        # Update epicycle positions and properties
        for i in range(min(num_terms, len(coeffs))):
            k, c_k = coeffs[i]
            radius = abs(c_k)
            phase = np.angle(c_k)

            # Update circle representation
            circles[i].center = (x, y)
            circles[i].radius = radius

            # Calculate next epicycle point
            next_x = x + radius * np.cos(k * t + phase)
            next_y = y + radius * np.sin(k * t + phase)

            # Update connecting line
            lines[i].set_data([x, next_x], [y, next_y])

            # Advance to next epicycle center
            x, y = next_x, next_y

        # Disable display of unused epicycles
        for i in range(min(num_terms, len(coeffs)), num_terms):
            circles[i].set_radius(0)
            lines[i].set_data([], [])

        # Update traced path
        traced_x.append(x)
        traced_y.append(y)
        path_line.set_data(traced_x, traced_y)

        # Update progress metrics
        frames_completed += 1
        progress_percent = (frames_completed / num_frames) * 100
        elapsed_time = time.time() - start_time

        if frames_completed > 1:
            estimated_total = elapsed_time * (num_frames / frames_completed)
            remaining_time = estimated_total - elapsed_time
            progress_text.set_text(f'Progress: {progress_percent:.1f}% | Est. remaining: {remaining_time:.1f}s')
        else:
            progress_text.set_text(f'Progress: {progress_percent:.1f}%')

        return [path_line, progress_text] + circles + lines

    # Initialize animation with specified parameters
    print("Generating animation frames...")
    ani = FuncAnimation(fig, update, frames=num_frames, init_func=init,
                        interval=1000 / fps, blit=True, repeat=True)

    plt.close()  # Prevent duplicate rendering in notebook environments
    print("✅ Animation setup complete!")
    return ani


def save_animation(ani, filename='smiley_fourier.gif', fps=20):
    """Save the animation to a high-quality file."""

    # Initialize progress tracking
    progress_bar = tqdm.tqdm(total=100, desc="Saving animation")

    # Setup progress monitoring callback
    last_frame = [0]

    def progress_callback(current_frame, total_frames):
        # Update progress display only on frame transitions
        if current_frame > last_frame[0]:
            progress_bar.update(100 * (current_frame - last_frame[0]) / total_frames)
            last_frame[0] = current_frame

    # Execute save operation with progress monitoring
    ani.save(filename, writer='pillow', fps=fps, dpi=200,
             progress_callback=progress_callback)

    progress_bar.close()
    print(f"✅ Animation saved successfully as {filename}!")


def main():
    print("🔹 Starting Fourier series smiley face animation generation 🔹")

    # Generate initial outline coordinates
    x, y = generate_smiley_outline()

    # Apply signal filtering for contour refinement
    print("Applying smoothing filters...")
    # Window size 15, polynomial order 3
    x = signal.savgol_filter(x, 15, 3)
    y = signal.savgol_filter(y, 15, 3)

    # Apply additional interpolation for enhanced smoothness
    print("Applying cubic interpolation for smoother curves...")
    t = np.linspace(0, 1, len(x))
    t_new = np.linspace(0, 1, 5000)  # Increased point density

    x = interpolate.interp1d(t, x, kind='cubic')(t_new)
    y = interpolate.interp1d(t, y, kind='cubic')(t_new)
    print("✅ Smoothing complete!")

    # Compute Fourier series representation
    coeffs = compute_fourier_coefficients(x, y, num_coeffs=200)

    # Generate animation
    ani = animate_fourier_drawing(coeffs, num_terms=150, duration_sec=15, fps=20)

    # Export animation
    print(f"Saving animation to smiley_fourier.gif...")
    print("This may take several minutes. Please be patient.")
    save_animation(ani, filename='../smiley_fourier.gif', fps=20)

    print("🎉 All done! The animation has been created successfully.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n❌ Process interrupted by user.")
    except Exception as e:
        print(f"\n❌ Error: {e}")
