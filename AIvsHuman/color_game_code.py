
#%%
# To install pyDOE, run the following command in your terminal or notebook:
import subprocess
import sys

# Install pyDOE if not already installed
try:
    import pyDOE
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pyDOE"])
    import pyDOE  # retry import after installing

import numpy as np
import matplotlib.pyplot as plt
import ipywidgets as widgets
from IPython.display import display, clear_output
import random
from scipy.stats import qmc
from scipy.spatial import distance
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from scipy.optimize import minimize
from scipy.stats import norm
from pyDOE import lhs
import os

class Color_Game():
    def __init__(self):
        self.seed = random.randint(0, 10000)  # Random seed for reproducibility
        np.random.seed(self.seed)  # Set seed for reproducibility
        random.seed(self.seed)
        os.environ["PYTHONHASHSEED"] = "0"  # For consistent hashing
                
        # Target color in RGB format
        #self.target_color = [75, 46, 131]
        self.target_color = (np.random.rand(3)*255).astype(int)  # Random target color
        self.target_color_norm = np.array(self.target_color)/255
        self.target_color_CMYK = self.RGB_to_CMYK(self.target_color)
        self.target_string = ', '.join(str(x) for x in self.target_color)

        # Optimization Parameters for Bayesian Optimization
        self.hyperparameter = 0.01
        self.threshold = 0.025
        self.scaling_factor = 50

        # Generate a row of random CMYK samples
        self.n_samples = 8
        self.max_well_volume = 250

        # Gaussian Process Regression model
        self.kernel = C(constant_value=1, constant_value_bounds=(1e-3, 1e3)) * RBF(length_scale=1, length_scale_bounds=(1e-3, 1e3))
        self.gp = GaussianProcessRegressor(kernel=self.kernel, n_restarts_optimizer=10, alpha=1e-4, normalize_y=False)
        # Initialize some functions
        self.build_widget_game()

    def uniform_sampling(self, num_rows):
        return np.random.uniform(0, 1, (num_rows, 4))

    def normal_sampling(self, num_rows):
        # Normal distribution samples are scaled to [0, 1] using the sigmoid function
        samples = np.random.normal(0, 1, (num_rows, 4))
        return 1 / (1 + np.exp(-samples))  # Sigmoid to map to [0, 1]

    def stratified_sampling(self, num_rows):
        strata = np.linspace(0, 1, 9)[:-1] + np.diff(np.linspace(0, 1, 9)) / 2
        samples = np.array([[random.choice(strata) for _ in range(4)] for _ in range(num_rows)])
        return samples

    def latin_hypercube_sampling(self, num_samples):
        # Using pyDOE for LHS, results are already in [0, 1] range
        return lhs(n=4, samples=num_samples)

    def latin_hypercube_sum_constraint(self,num_samples, sum_constraint):
        lhs_samples = lhs(4, samples=num_samples, criterion='maximin')

        # Adjust each sample to meet the sum constraint
        for i in range(num_samples):
            current_sum = np.sum(lhs_samples[i, :])
            lhs_samples[i, :] *= sum_constraint / current_sum

        return lhs_samples

    def sobol_sampling(self, num_rows):
        sampler = qmc.Sobol(d=4, scramble=True)
        # Scale the samples to be in the range [0, 1]
        return qmc.scale(sampler.random_base2(m=int(np.log2(num_rows))), 0, 1)

    def halton_sampling(self, num_rows):
        sampler = qmc.Halton(d=4, scramble=True)
        return qmc.scale(sampler.random(n=num_rows), 0, 1)

    # Function to convert CMYK to RGB
    def CMYK_to_RGB(self, c, m, y, k):
        R = (1 - c) * (1 - k)
        G = (1 - m) * (1 - k)
        B = (1 - y) * (1 - k)
        return np.array([R, G, B])

    def RGB_to_CMYK(self, RGB_sample):
        Red = RGB_sample[0]/255
        Green = RGB_sample[1]/255
        Blue = RGB_sample[2]/255

        Black = min(1-Red,1-Green,1-Blue)
        Cyan = (1-Red-Black)/(1-Black)
        Magenta = (1-Green-Black)/(1-Black)
        Yellow  = (1-Blue-Black)/(1-Black)

        return np.array([Cyan, Magenta, Yellow, Black])

    def plot_sampling(self):
        # Visualize the color combinations chosen by the algorithm over iterations.
        if (self.n_samples+1) / 10 > 1:
            colors_per_row = 10  # Limit to 10 colors per row
        else:
            colors_per_row = self.n_samples + 1

        n_rows = (self.n_samples + colors_per_row) // colors_per_row  # Calculate total rows

        plt.figure(figsize=(colors_per_row, n_rows * 1.4))
        plt.suptitle('Random Samples')

        for idx, rgb in enumerate(self.random_RGB):
            row = idx // colors_per_row
            col = idx % colors_per_row
            # Each color is displayed in its own grid
            plt.subplot(n_rows, colors_per_row, idx + 1)
            plt.imshow([[rgb]], extent=[0, 1, 0, 1])
            plt.axis("off")
            plt.title(f"Sample {idx + 1}", fontsize=8)

        plt.tight_layout()

    # Calculate the 2-norm distance for each sample color
    def norm_error(norm_samples, target_color):
        return [np.linalg.norm(np.array(sample) - np.array(target_color)/255) for sample in norm_samples]

    def plot_norm_error(self):
        sample_distances = self.norm_error(self.random_RGB, self.target_color)

        # Plot the distances
        plt.figure(figsize=(8, 6))
        plt.scatter(range(len(self.random_RGB)), sample_distances, marker='o')
        plt.title('2-Norm Distance from Target RGB Color of LHS')
        plt.xlabel('Sample Index')
        plt.ylabel('2-Norm Distance')
        plt.ylim(0,1)
        plt.grid(True)
        plt.show()


    #%%
    # Expected Improvement maximizes implicetly
    def expected_improvement(self, X, X_sample, y_sample, model):
        mu, sigma = model.predict(X, return_std=True)

        optiminal_sample = np.min(y_sample)

        with np.errstate(divide="ignore"):

            imp = mu - optiminal_sample - self.hyperparameter
            Z = imp / sigma
            ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
            ei[sigma == 0.0] = 0.0
        return ei.flatten()

    # Function to propose the next sample point using Expected Improvement
    def propose_location(self, acquisition, X_sample, y_sample, bounds, n_restarts=25):
        dim = X_sample.shape[1]

        def min_obj(x):
            return -acquisition(x.reshape(-1, dim), X_sample, y_sample, self.gp)

        # Starting points for optimization
        x0_list = np.random.uniform(0, 1, size=(n_restarts, dim))
        best_x, best_val = None, float("inf")
        for x0 in x0_list:
            res = minimize(min_obj, x0, bounds=bounds, method="L-BFGS-B")
            if res.fun < best_val:
                best_val = res.fun
                best_x = res.x
        return best_x

    def bayesian_optimization(self, random_samples, max_iterations, threshold = 5):

        # Initialize variables
        X = random_samples
        y = -np.linalg.norm(X - self.target_color_CMYK, axis=1)

        n_features = n_features = X[0].shape[0]  # CMYK components
        bounds = [(0, 1)] * n_features  # CMYK values are normalized between 0 and 1

        # Run Bayesian Optimization and visualize
        distances = []  # Track distances to the target
        X_samples = []  # Start with the first sample
        y_samples = []

        for i in range(max_iterations):
            self.gp.fit(X, y)  # Fit GP model

            # Propose the next sampling point
            X_next = self.propose_location(self.expected_improvement, X, y, bounds)

            # Evaluate the objective function for the new sample
            y_next = -np.linalg.norm(X_next - self.target_color_CMYK)
            distances.append(-y_next)  # Track the negative distance (closer to target)

            # Append the new sample to the dataset
            X = np.vstack((X, X_next))
            y = np.append(y, y_next)

            # Track the selected color
            X_samples.append(X_next)
            y_samples.append(y_next)

            # Check if the error is below the threshold
            if -y_next < threshold/ self.scaling_factor:
                print(f"Optimization stopped early: error below {threshold} at iteration {i + 1}")
                break

        return X_samples, y_samples

    #%%
    def build_widget_game(self):
        # Initialize storage for attempts
        self.user_attempts = []
        self.cmyk_attempts = []
        self.bo_guesses = []
        self.user_distances = []
        self.bo_distances = []
        self.X_samples = []
        self.y_samples = []
        self.threshold = 5

        # Initialize variables
        n_features = 4  # CMYK components
        self.bounds = [(0, 1)] * n_features  # CMYK values are normalized between 0 and 1
        self.random_CMYK = (self.latin_hypercube_sum_constraint(self.n_samples, self.max_well_volume)/self.max_well_volume)
        self.random_RGB = np.array([self.CMYK_to_RGB(*cmyk) for cmyk in self.random_CMYK])  # Convert to RGB


        self.X = (self.latin_hypercube_sum_constraint(self.n_samples, self.max_well_volume)/self.max_well_volume)
        self.y = -np.linalg.norm(self.X - self.target_color_CMYK, axis=1)

        # Gaussian Process Regression model
        kernel = C(constant_value=1, constant_value_bounds=(1e-3, 1e3)) * RBF(length_scale=1, length_scale_bounds=(1e-3, 1e3))
        self.gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, alpha=1e-4, normalize_y=False)

        # Create interactive sliders
        self.c_slider = widgets.FloatSlider(description="Cyan (%)", min=0, max=1, step=0.01, value=0.5, continuous_update=False)
        self.m_slider = widgets.FloatSlider(description="Magenta (%)", min=0, max=1, step=0.01, value=0.5, continuous_update=False)
        self.y_slider = widgets.FloatSlider(description="Yellow (%)", min=0, max=1, step=0.01, value=0.5, continuous_update=False)
        self.k_slider = widgets.FloatSlider(description="Black (%)", min=0, max=1, step=0.01, value=0.5, continuous_update=False)
        self.submit_button = widgets.Button(description="Submit")

        # Title Widget
        self.title = widgets.HTML("<h2>Color Optimization: Minimize Error to Reach Target Color</h2>")

        # Output widgets for displaying colors and attempts
        self.user_output = widgets.Output()
        self.bo_output = widgets.Output()
        self.random_output = widgets.Output()
        self.attempt_tracker = widgets.IntText(value=0, description="Attempts:", disabled=True)
        self.min_distance_display = widgets.FloatText(value=float("inf"), description="Best Guess:", disabled=True)
        self.winner_display = widgets.HTML(value="<h2 style='text-align:center; font-size: 20px;'>Goal: < 5% Error before BO</h2>")
        self.error_output = widgets.Output()
        self.gp_plot_output = widgets.Output()
        self.ei_plot_output = widgets.Output()

    # Function to update the display on button click
    def submit_color(self,_):
        # Initial the game
        #global user_attempts, cmyk_attempts, X, y, bo_guesses, bo_distances, gp

        # Convert user CMYK input to RGB
        cmyk_values = (self.c_slider.value,self. m_slider.value, self.y_slider.value, self.k_slider.value)
        user_rgb = (self.CMYK_to_RGB(*cmyk_values) * 255).astype(int)

        # Compute Euclidean distance from the target color
        distance = np.linalg.norm(cmyk_values - self.target_color_CMYK) * self.scaling_factor

        if np.isnan(distance) or np.isinf(distance):
            distance = 0.0

        # Store attempt data
        self.user_attempts.append(user_rgb)
        self.cmyk_attempts.append(cmyk_values)
        self.user_distances.append(distance)

        # Update attempt count
        self.attempt_tracker.value = len(self.user_attempts)

        # Update the minimum distance
        self.min_distance_display.value = round(min(self.user_distances), 1) if self.user_distances else None

        # Only run BO if the best distance is greater than the threshold
        if not self.bo_distances or min(self.bo_distances) > 0.05:
            # Update GP model
            self.gp.fit(self.X, self.y)  # Fit GP model to the current data

            # Propose the next sampling point based on Bayesian Optimization
            X_next = self.propose_location(self.expected_improvement, self.X, self.y, self.bounds)
            y_next = -np.linalg.norm(X_next - self.target_color_CMYK)  # Compute the negative distance

            # Append the new sample to the dataset
            self.X_samples.append(X_next)
            self.y_samples.append(y_next)

            self.X = np.vstack((self.X, X_next))
            self.y = np.append(self.y, y_next)

            # Track the selected color
            bo_rgb = (self.CMYK_to_RGB(X_next[0],X_next[1],X_next[2],X_next[3])*255).astype(int)
            self.bo_distances.append(np.linalg.norm(X_next - self.target_color_CMYK)*self.scaling_factor) # Track the negative distance (closer to target)
            self.bo_guesses.append(X_next)

        # Determine the winner
        if any(d < 5 for d in self.user_distances):
            self.winner_display.value = "<h2 style='text-align:center; font-size: 24px;'>User Wins 💪🏻 🧑🏻!</h2>"
        elif any(d < 5 for d in self.bo_distances):
            self.winner_display.value = "<h2 style='text-align:center; font-size: 24px;'>AI Wins! 🦾 🤖 </h2>"
        else:
            self.winner_display.value = "<h2 style='text-align:center; font-size: 20px;'>Goal: < 5% Error before BO</h2>"

        # Clear output and redraw user attempts
        with self.user_output:
            clear_output(wait=True)

            # Determine the number of rows needed for wrapping
            num_attempts = len(self.user_attempts)
            num_cols = 8  # Max colors per row
            num_rows = -(-num_attempts // num_cols)  # Ceiling division

            fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, num_rows * 2.1), constrained_layout=True)
            axes = np.array(axes).reshape(num_rows, num_cols)  # Ensure correct shape

            # Plot attempts with CMYK labels and distances
            for i, (rgb, cmyk, dist) in enumerate(zip(self.user_attempts, self.cmyk_attempts, self.user_distances)):
                row, col = divmod(i, num_cols)
                axes[row, col].imshow([[rgb]])
                axes[row, col].set_title(
                    f"C:{cmyk[0]:.2f}% M:{cmyk[1]:.2f}%\nY:{cmyk[2]:.2f}% K:{cmyk[3]:.2f}%\nError:{dist:.1f}%", fontsize=7)
                axes[row, col].axis("off")

            # Hide unused subplots
            for i in range(num_attempts, num_rows * num_cols):
                row, col = divmod(i, num_cols)
                axes[row, col].axis("off")

            plt.suptitle("User Attempts", fontsize=12)
            plt.show()

        # Clear output and redraw BO guesses
        with self.bo_output:
            clear_output(wait=True)

            num_bo_guesses = len(self.bo_guesses)
            num_cols = 8
            num_rows = -(-num_bo_guesses // num_cols)

            fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, num_rows * 2.1), constrained_layout=True)
            axes = np.array(axes).reshape(num_rows, num_cols)

            # Plot BO guesses
            for i, (bo_guess, dist) in enumerate(zip(self.bo_guesses, self.bo_distances)):
                row, col = divmod(i, num_cols)
                bo_rgb = self.CMYK_to_RGB(*bo_guess)
                bo_rgb_255 = (bo_rgb * 255).astype(int)
                axes[row, col].imshow([[bo_rgb]])
                axes[row, col].set_title(f"R:{bo_rgb_255[0]} G:{bo_rgb_255[1]}\n B:{bo_rgb_255[2]}\nError:{dist:.1f}%", fontsize=7)
                axes[row, col].axis("off")

            # Hide extra subplots
            for i in range(num_bo_guesses, num_rows * num_cols):
                row, col = divmod(i, num_cols)
                axes[row, col].axis("off")

            plt.suptitle("BO Attempts", fontsize=12)
            plt.show()

        # # Plot convergence
        # with self.error_output:
        #     clear_output(wait=True)  # Clear previous output before displaying the new plot
        # # Plot the convergence
        #     plt.figure(figsize=(8, 6))
        #     plt.plot(range(1, len(self.bo_distances) + 1), self.bo_distances, marker='o', linestyle='-', color='b')
        #     plt.title('Convergence of Bayesian Optimization')
        #     plt.xlabel('Iteration')
        #     plt.ylabel('Distance to Target Color')
        #     plt.grid(True)
        #     plt.show()

        # # Plot the training data, predicted mean, and confidence interval
        # with self.gp_plot_output:
        #     clear_output(wait=True)

        #     n_features = self.X_samples[0].shape[0]  # Should be 4 for CMYK
        #     fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        #     for i in range(n_features):
        #         # Fix other features to their mean values from training data
        #         #fixed_values = np.mean(X_samples, axis=0)
        #         fixed_values = self.target_color_CMYK.copy()

        #         # Generate a grid for the ith feature
        #         X_grid = np.linspace(0, 1, 100).reshape(-1, 1)
        #         X_full_grid = np.tile(fixed_values, (100, 1))
        #         X_full_grid[:, i] = X_grid.ravel()  # Vary only the ith feature

        #         # Predict the mean and standard deviation
        #         mu, sigma = self.gp.predict(X_full_grid, return_std=True)

        #         # Convert lists to numpy arrays if necessary
        #         mu = np.array(mu)
        #         sigma = np.array(sigma)
        #         y_samples_arr = np.array(self.y_samples)

        #         # Calculate the true objective (2-norm distance to target color)
        #         true_objective = [np.linalg.norm(cmyk - self.target_color_CMYK) for cmyk in X_full_grid]

        #         ax = axes[i // 2, i % 2]

        #         # Plot the true objective function
        #         ax.plot(X_grid, true_objective, color='black', label='True Objective')

        #         # Plot the predicted mean
        #         ax.plot(X_grid, -mu, color='blue', linestyle='--', label='Predicted Mean')
        #         ax.fill_between(
        #             X_grid.ravel(), -mu - 1.96 * sigma, -mu + 1.96 * sigma,
        #             color='blue', alpha=0.2, label='95% Confidence Interval'
        #         )

        #         # Plot the training data for this feature
        #         ax.scatter([x[i] for x in self.X_samples], -y_samples_arr, color='red', zorder=10, label='Training Data')

        #         ax.scatter(X_next[i], -y_next, color='green', zorder=10, label='Next Sample')

        #         # Labels and title
        #         ax.set_title(f"GP Regression for Feature {['C', 'M', 'Y', 'K'][i]}", fontsize=14)
        #         ax.set_xlabel(f"{['C', 'M', 'Y', 'K'][i]} Value", fontsize=12)
        #         ax.set_ylabel("Distance to Target", fontsize=12)
        #         ax.legend()

        #     plt.tight_layout()
        #     plt.show()

        # # Plot expected improvement
        # with self.ei_plot_output:
        #     clear_output(wait=True)

        #     n_features = len(self.X_samples[0])  # Number of features (should be 4 for CMYK)
        #     fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        #     for i, ax in enumerate(axes.flatten()):
        #         # Fix other features to their mean values from training data
        #         fixed_values = np.mean(self.X_samples, axis=0)

        #         # Generate a grid for the ith feature
        #         X_grid = np.linspace(0, 1, 100).reshape(-1, 1)
        #         X_full_grid = np.tile(fixed_values, (100, 1))
        #         X_full_grid[:, i] = X_grid.ravel()  # Vary only the ith featur

        #         # Compute EI
        #         ei = self.expected_improvement(X_full_grid, np.array(self.X_samples), np.array(self.y_samples), self.gp)

        #         # Find the next sampling point (max EI)
        #         next_index = np.argmax(ei)
        #         next_x = X_grid[next_index]
        #         next_ei = ei[next_index]

        #         # Plot EI curve
        #         ax.plot(X_grid, ei, color='purple', linestyle='-', label='Expected Improvement')

        #         # Highlight next sampling point
        #         ax.scatter(next_x, next_ei, color='red', s=100, marker='o', label='Next Sample Point')
        #         ax.axvline(next_x, color='red', linestyle='--', alpha=0.5)

        #         # Titles and labels
        #         ax.set_title(f"Expected Improvement for Feature {['C', 'M', 'Y', 'K'][i]}")
        #         ax.set_xlabel(f"{['C', 'M', 'Y', 'K'][i]} Value")
        #         ax.set_ylabel("Expected Improvement")
        #         ax.legend()
        #         ax.grid(True)

        #     plt.tight_layout()
        #     plt.show()

    # Attach the button click event
    def launch_game(self):
        self.submit_button.on_click(self.submit_color)

        # Display target color and attempt tracker immediately
        target_display = widgets.Output()
        with target_display:
            fig, ax = plt.subplots(figsize=(2, 2))
            ax.imshow([[self.target_color]])
            ax.axis("off")
            plt.title("Target Color (" + self.target_string + ")")
            plt.show()

        # Layout: Sliders + Target Color + Attempt Tracker + Min Distance in the same row
        control_panel = widgets.HBox([widgets.VBox([self.c_slider, self.m_slider, self.y_slider, self.k_slider, self.submit_button]),
                                    target_display,
                                    widgets.VBox([self.attempt_tracker, self.min_distance_display, self.winner_display])])

        # Show the initial random sampling row (remains visible)
        with self.random_output:
            fig, axes = plt.subplots(1, 8, figsize=(15, 2))
            for i, (rgb, cmyk) in enumerate(zip(self.random_RGB, self.random_CMYK)):
                axes[i].imshow([[rgb]])
                axes[i].set_title(f"C:{cmyk[0]:.2f}% M:{cmyk[1]:.2f}% \n Y:{cmyk[2]:.2f}% K:{cmyk[3]:.2f}%", fontsize=7)
                axes[i].axis("off")

            plt.suptitle("Random Samples", fontsize=12)
            plt.tight_layout()
            plt.show()

        # Display everything
        display(widgets.VBox([self.title, control_panel, self.user_output, self.bo_output, self.random_output]))
    # %%
