import numpy as np
from sklearn.linear_model import LinearRegression, RANSACRegressor, TheilSenRegressor, HuberRegressor
from sklearn.metrics import r2_score
from scipy.stats import mstats
import matplotlib.pyplot as plt


def generate_random_data(a, b, num_samples=100, noise_type='normal', noise_params=None):
    """
    Generate random data based on the linear function ax + by = value with specified noise,
    ensuring value(x, y) >= 100(x + y).

    Parameters:
    a (float): Coefficient for x.
    b (float): Coefficient for y.
    num_samples (int): Number of samples to generate.
    noise_type (str): Type of noise to add ('normal', 'cauchy', 'laplace').
    noise_params (dict): Parameters for the noise distribution.

    Returns:
    data (list of tuples): Generated data as a list of tuples (x, y, value).
    """
    if noise_params is None:
        noise_params = {}

    np.random.seed(0)
    x = np.random.uniform(1, 20, num_samples)
    y = np.random.uniform(1, 20, num_samples)

    if noise_type == 'normal':
        # Normal distribution noise
        noise_mean = noise_params.get('mean', 0)
        noise_std = noise_params.get('std', 1)
        noise = np.random.normal(noise_mean, noise_std, num_samples)
    elif noise_type == 'cauchy':
        # Cauchy distribution noise
        noise_loc = noise_params.get('loc', 0)
        noise_scale = noise_params.get('scale', 1)
        noise = np.random.standard_cauchy(num_samples) * noise_scale + noise_loc
    elif noise_type == 'laplace':
        # Laplace distribution noise
        noise_loc = noise_params.get('loc', 0)
        noise_scale = noise_params.get('scale', 1)
        noise = np.random.laplace(noise_loc, noise_scale, num_samples)
    else:
        raise ValueError("Unsupported noise type. Use 'normal', 'cauchy', or 'laplace'.")

    value = a * x + b * y + noise

    # Ensure that value(x, y) >= 100(x + y)
    min_price_per_gas_unit = 100
    min_value = min_price_per_gas_unit * (x + y)
    value = np.maximum(value, min_value)

    data = list(zip(x, y, value))
    return data


def fit_model_from_data(data, method='linear'):
    """
    Fit a regression model based on the provided data and method.

    Parameters:
    data (list of tuples): Input data as a list of tuples (weight_x, weight_y, tot_value).
    method (str): The regression method to use. Options are 'linear', 'ransac', 'theil_sen', 'huber'.

    Returns:
    model: The fitted regression model.
    coefficients (array): Coefficients of the fitted model.
    """

    # Step 1: Convert data to numpy arrays
    weights_x = np.array([x[0] for x in data])
    weights_y = np.array([x[1] for x in data])
    tot_values = np.array([x[2] for x in data])

    # Stack weights for the input matrix
    X = np.column_stack((weights_x, weights_y))
    y = tot_values

    # Step 2: Fit the model based on the chosen method
    if method == 'linear':
        model = LinearRegression()
        model.fit(X, y)
        coefficients = model.coef_
    elif method == 'ransac':
        # RANSAC Parameters:
        # base_estimator: the model to fit (LinearRegression in this case)
        # min_samples: minimum number of samples to fit the model (50% of total samples here)
        # residual_threshold: residual threshold to determine inliers (in absolute terms -- not proportional terms)
        # max_trials: maximum number of iterations the algorithm attempts to find a valid model
        base_model = LinearRegression()
        model = RANSACRegressor(base_estimator=base_model, min_samples=0.5, residual_threshold=20.0, max_trials=100)
        model.fit(X, y)
        coefficients = model.estimator_.coef_
    elif method == 'theil_sen':
        # TheilSenRegressor is robust to outliers and does not require parameter tuning
        model = TheilSenRegressor()
        model.fit(X, y)
        coefficients = model.coef_
    elif method == 'huber':
        # HuberRegressor combines the properties of both least squares and absolute error loss functions
        model = HuberRegressor()
        model.fit(X, y)
        coefficients = model.coef_
    else:
        raise ValueError("Method not recognized. Use 'linear', 'ransac', 'theil_sen', or 'huber'.")

    return model, coefficients


def disperse_value_per_unit(data, coefficients):

    a, b = coefficients
    contributions_per_unit = []

    for (x, y, value) in data:
        denominator = a * x + b * y
        contribution_per_unit_x = (a / denominator) * value
        contribution_per_unit_y = (b / denominator) * value
        contributions_per_unit.append((contribution_per_unit_x, contribution_per_unit_y))

    return contributions_per_unit

def winsorize_data(data, limits=(0.05, 0.05)):
    """
    Winsorize the data to limit the influence of extreme values.

    Parameters:
    data (array): The data to Winsorize.
    limits (tuple): The fraction of data to Winsorize from each end.

    Returns:
    winsorized_data (array): The Winsorized data.
    """
    winsorized_data = mstats.winsorize(data, limits=limits)
    return winsorized_data

def plot_histogram(data, title, xlabel, ylabel, bins):
    """
    Plot a histogram for the given data.

    Parameters:
    data (array): The data to plot.
    title (str): The title of the histogram.
    xlabel (str): The label for the x-axis.
    ylabel (str): The label for the y-axis.
    bins (int): The number of bins for the histogram.

    Returns:
    None
    """
    plt.figure(figsize=(10, 6))
    plt.hist(data, bins=bins, edgecolor='k', alpha=0.7)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.show()


# Example usage

# Generate example data
a = 200
b = 300
num_samples = 200

# Define noise parameters for different noise types
noise_params_normal = {'mean': 0, 'std': 50}
noise_params_cauchy = {'loc': 0, 'scale': 50}
noise_params_laplace = {'loc': 0, 'scale': 50}

# Generate data with different noise types
data_normal = generate_random_data(a, b, num_samples, noise_type='normal', noise_params=noise_params_normal)
data_cauchy = generate_random_data(a, b, num_samples, noise_type='cauchy', noise_params=noise_params_cauchy)
data_laplace = generate_random_data(a, b, num_samples, noise_type='laplace', noise_params=noise_params_laplace)

# Fit model
data = data_cauchy
chosen_method = 'huber'  # Change to 'linear', 'theil_sen', 'huber', 'ransac' as needed
model, coefficients = fit_model_from_data(data, method=chosen_method)

print(f"Method: {chosen_method}")
print("Noise: " + "cauchy")
print("Coefficients:", coefficients)

# Convert data to numpy arrays for evaluation
weights_x = np.array([x[0] for x in data])
weights_y = np.array([x[1] for x in data])
data_values = np.array([x[2] for x in data])
X = np.column_stack((weights_x, weights_y))


# Evaluate model performance using R²
predicted_values = model.predict(X)
r2 = r2_score(predicted_values, data_values)

print("R² Score:", r2)

# Disperse the value based on the fitted model coefficients
contributions = disperse_value_per_unit(data, coefficients)

# Get statistics
# Extract contributions for X and Y
contributions_x = np.array([c[0] for c in contributions])
contributions_y = np.array([c[1] for c in contributions])

# Winsorize the contributions. The limits are the top and bottom fractions to be capped.
winsorized_x = winsorize_data(contributions_x, limits=(0.01, 0.01))
winsorized_y = winsorize_data(contributions_y, limits=(0.01, 0.01))

# Determine the number of bins for 1% bin size
range_x = winsorized_x.max() - winsorized_x.min()
range_y = winsorized_y.max() - winsorized_y.min()
bin_size_x = range_x * 0.01
bin_size_y = range_y * 0.01
num_bins_x = int(range_x / bin_size_x)
num_bins_y = int(range_y / bin_size_y)

# Plot histograms for the Winsorized contributions
plot_histogram(winsorized_x, 'Histogram of Winsorized Contributions for X', 'Contribution X', 'Frequency', bins=num_bins_x)
plot_histogram(winsorized_y, 'Histogram of Winsorized Contributions for Y', 'Contribution Y', 'Frequency', bins=num_bins_y)

# Calculate Winsorized medians
winsorized_median_x = np.median(np.asarray(winsorized_x))
winsorized_median_y = np.median(np.asarray(winsorized_y))

# Print the Winsorized medians
print("Winsorized Median for X:", winsorized_median_x)
print("Winsorized Median for Y:", winsorized_median_y)