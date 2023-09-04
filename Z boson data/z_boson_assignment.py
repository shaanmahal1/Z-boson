
"""
Title: PHYS20161 Final Assignment Z^0 Boson

This code calculates the mass, width and lifetime of the Z^0 boson by reading
in data from electron-positron collisions. These files are .csc files. The
code also evaluates the uncertainties of the fitting parameters as well as the
reduced chi value. Moreover, the code plots the experimental data with the
cross section function fitted to the data.

Shaan Mahal 26/11/21
"""
import sys
import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as pc
from scipy.optimize import curve_fit

DATA_1= 'z_boson_data_1.csv'
DATA_2 = 'z_boson_data_2.csv'

#units in GeV
GAMMA_EE = 83.91*10**-3

# function definitions
def full_data(data_1, data_2):
    """
    This reads in the two datasets and combines them vertically. This includes
    a file check at the start to make sure the file can be opened.

    Parameters
    ----------
    data_1 : numpy array of floats
    data_2 : numpy array of floats

    Returns
    ----------
    total_data : numpy array of floats
    """

    #file check
    try:
        dataset_1 = np.genfromtxt(data_1 , delimiter = ',', comments = 'fail',
                                  skip_header = 1)
    except IOError:
        print("File '{0}' could not open".format(data_1))
        sys.exit()

    try:
        dataset_2 = np.genfromtxt(data_2, delimiter = ',', comments = '%')
    except IOError:
        print("File '{0}' could not open".format(data_2))
        sys.exit()

    total_data = np.vstack((dataset_1, dataset_2))

    return total_data

def filtered_data(total_data):
    """
    This filters the data of any errors and nan values. This also
    removes any outliers within 3 standard deviations of the mean and removes
    values less than zero.

    Parameters
    ----------
    total_data : numpy array of floats

    Returns
    ----------
    total_data : 2 dimensional array of floats
    """

    total_data = total_data[~np.isnan(total_data).any(axis=1)]
    mean = np.mean(total_data[:,1])
    standard_deviation = np.std(total_data[:,1])
    total_data = total_data[abs(total_data[:,1] - mean) < 3*standard_deviation]
    total_data = total_data[np.where(total_data[:,1] > 0)]
    total_data = total_data[np.where(total_data[:,2] > 0)]

    return total_data

def is_outliers(total_data, mass, width):
    """
    This filters the remaining outliers contained within the data.

    Parameters
    ----------
    total_data : 2 dimensional array of floats
    mass : float
    width : float

    Returns
    -------
    total_data : 2 dimensional array of floats
    """

    total_data = total_data[(abs(cross_section_function(total_data[:,0], mass,
                 width) - (total_data[:,1])))<(3*(total_data[:,2]))]

    return total_data

def cross_section_function(energy, mass, width):
    """
    This is the function that is fitted to the data points. This function
    returns the cross section.

    Parameters
    ----------
    energy : array of floats
    mass : float
    width : float

    Returns
    -------
    cross section : function
    """

    return ((12 * np.pi) / mass**2) * \
           (energy**2 / ((energy**2 - mass**2)**2 + \
           (mass * width)**2)) * GAMMA_EE**2 * (0.3894 *10**6)

def plot_function(total_data):
    """
    This plots the experimental data points aswell as the fit.

    Parameters
    ----------
    total_data : 2 dimensional array of floats

    Returns
    -------
    total_data : 2 dimensional array of floats
    popt: array of floats
    pcov: covariance matrix of floats
    """

    fig = plt.figure()
    axis = fig.add_subplot(111)

    axis.set_title('Cross Section vs Energy', fontsize=18)
    axis.set_xlabel('Energy / GeV', fontsize=14)
    axis.set_ylabel('Cross Section / nb', fontsize=14)
    x_values = total_data[:,0]
    y_values = total_data[:,1]
    plt.scatter(x_values, y_values, color = 'c' , label='Experimental Data')
    axis.errorbar(x_values, y_values, yerr = total_data[:,2], ls='none',
                  fmt='black')
    initial_guess = [90,3]
    #pylint: disable=unbalanced-tuple-unpacking
    popt, pcov = (curve_fit(cross_section_function, x_values, y_values,
                 initial_guess))
    x_fit = np.linspace(x_values.min(), x_values.max(), 150)
    plt.plot(x_fit, (cross_section_function(x_fit, *popt)), 'blue',
             label = 'Fit')
    plt.savefig('Cross Section vs Energy', dpi = 300)

    axis.legend()
    plt.show()

    return total_data, popt, pcov

def fit(x_values, y_values):
    """
    This performs a fit of the data by using curve_fit.

    Parameters
    ----------
    x_values : floats
    y_values : floats

    Returns
    -------
    popt : array of floats
    pcov : covariance matrix of floats
    """

    initial_guess = [90,3]
    #pylint: disable=unbalanced-tuple-unpacking
    popt, pcov = (curve_fit(cross_section_function, x_values, y_values,
                 initial_guess))

    return popt, pcov

def get_reduced_chi_square(total_data, mass, width):
    """
    This measures the reduced chi square from using the chi square and
    degrees of freedom.

    Parameters
    ----------
    total_data : 2 dimensional array of floats
    mass : float
    width : float

    Returns
    -------
    reduced chi_square : float
    """

    degrees_of_freedom = len(total_data) - 2
    expected = cross_section_function(total_data[:,0], mass, width)
    chi_square = np.sum(((expected - total_data[:,1])**2 / total_data[:,2]**2))
    # returns reduced chi square
    return chi_square / degrees_of_freedom

def lifetime_function(width):
    """
    This function calculates the lifetime of the Z boson.

    Parameters
    ----------
    width : float

    Returns
    -------
    lifetime: float
    """

    return pc.hbar / (width* 10**9 * pc.electron_volt)

def main_function():
    """
    This is the main function that recalls all other functions and prints
    the calculations.

    Returns
    -------
    0 : int
    """
    total_data = full_data(DATA_1, DATA_2)
    total_data = filtered_data(total_data)
    x_values = total_data[:,0]
    y_values = total_data[:,1]
    popt, pcov = fit(x_values, y_values)
    mass, width = popt
    total_data= is_outliers(total_data, mass, width)
    total_data, popt, pcov = plot_function(total_data)
    reduced_chi = get_reduced_chi_square(total_data, mass, width)
    mass_uncertainty = np.sqrt(pcov[0,0])
    width_uncertainty = np.sqrt(pcov[1,1])
    lifetime = lifetime_function(width)
    lifetime_uncertainty = (width_uncertainty / width) * lifetime
    print('The mass of the Z boson is: {0:.4g}'.format(mass), \
          '± {0:.4g}'.format(mass_uncertainty), 'GeV /c^2')
    print('The width of the Z boson is: {0:.4g}'.format(width), \
          '± {0:.4g}'.format(width_uncertainty), 'GeV')
    print('The lifetime of the Z boson is: {0:.3g}'.format(lifetime), \
          '± {0:.3g}'.format(lifetime_uncertainty), 's')
    print('The reduced chi^2 value is: {0:.3f}'.format(reduced_chi))

    return 0
main_function()
