# dalec2
DALEC2 model and 4D-Var assimilation functions. 
# ------------------------------------------------------------------------------
# DALEC2 model (Bloom and Williams 2015)
# Code by Ewan Pinnington: ewan.pinnington@gmail.com
# ------------------------------------------------------------------------------

This project is written in object oriented Python, using classes to extract data from a csv file and run the DALEC2
model.


DATA:
data_class.py provides a class which extracts data from a csv (currently using observations from Alice Holt flux site in
Hampshire) for use with the DALEC2 model. To create a data class in IPython we would navigate to the directory
containing data_class.py and then type:

import data_class as dc

d = dc.DalecData(1999, 2000, 'nee')

Here this is saying we want to use data from the start of 1999 to the beginning of 2000 and we want to use observation
of Net Ecosystem Exchange ('nee') for assimilation. Currently only observation of NEE are available for assimilation.


MODEL:
Once we have imported and created our data class we next want to setup our model to assimilated observations or simply
perform a model run. to do this we type from the same directory in IPython:

import mod_class as mc

m = mc.DalecModel(d)

This sets up the model to run with the data we imported in our data class (d). There are many functions within this
model class that can now be run. For example if we want to run our model over the specified data in the data class we
would type:

model_output = m.mod_list(d.xb)

This would run the DALEC2 model over the years data we extracted with our data class for the initial conditions d.xb,
where d.xb is an initial guess to the model state and parameters for our site (This is an array of size 23 corresponding
to the 17 model parameters and 6 initial state members).


ASSIMILATION:
Here we are performing 4D-Var data assimilation for parameter and state estimation (for more details please see the
paper in Agricultural and Forest Meteorology by Pinnington et al. 2016). To set up an assimilation we would first setup
our data class for the period and observations we wished to assimilate, then create our model class using this data
class and finally run the assimilation as follows:

import data_class as dc

d = dc.DalecData(1999, 2000, 'nee')

import mod_class as mc

m = mc.DalecModel(d)

assimilation_results = m.find_min_tnc(d.xb)

So here we have decided on a single years assimilation window with observations of NEE, then created our model class
using this data and then started an assimilation where we have specified d.xb as our prior model guess to the initial
state and parameters of the system. This will then return the results from our minimiztion using the observations and
prior information, giving us the minimum of the cost function or the analysis (xa), which provide the best fit to the
assimilated observation and prior information. The form of the returned results is a tuple containing the analysis xa
the number of funtion iterations to converge to the minimum and the pass code for the minimum (xa, iterations, code).
For more information please see, http://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.optimize.fmin_tnc.html


PLOTTING:
In order to plot the results we find from the assimilation as a time series of assimilation and forecast we use
functions from the plot.py file. If we continue on from the previous example we have the following:

import plot as p

d2 = dc.DalecData(1999, 2013, 'nee')

ax, fig = p.plot_4dvar('nee', d2, xb=d.xb, xa=assimilation_results[0], awindl=d.len_ren)

fig.savefig('~/example_directory/nee_assimilation.pdf')

This will save our plot to an example directory as a pdf. Here the prior or background model trajectory will be
displayed as a blue line, the analysis or posterior model trajectory a green line and the observations will be orange
dots in both the assimilation window and the forecast period (before and after the dotted line). If we just want to
display the plot in python we can type the following:

ax, fig = p.plot_4dvar('nee', d2, xb=d.xb, xa=assimilation_results[0], awindl=d.len_ren)

import matplotlib.pyplot as plt

plt.show()

This will bring up the plot in the python sessions. There are a few other plotting functions in the plot.py file for
plotting observation and data time series and assimilation results, for more information please read through the
plotting function doc strings.


Control Variable Transform Assimilation:
As the assimilation and manipulation of large matrices can become ill-conditioned for larger problems, we have also
implemented a control variable transform assimilation routine to pre-condition the problem. This is run in much the
same way as the original assimilation. However an extra argument can be called that will pickle the assimilation results
as a dictionary to a specified file. To run this we would type the following:

import data_class as dc

d = dc.DalecData(1999, 2000, 'nee')

import mod_class as mc

m = mc.DalecModel(d)

file_name_to_pickle_results = '~/example_directory/assimilation_results_nee_99_00.p'

assimilation_results_cvt, xa = m.find_min_tnc_cvt(d.xb, f_name=file_name_to_pickle_results)

We could then again plot the assimilation time series by typing:

import plot as p

d2 = dc.DalecData(1999, 2013, 'nee')

ax, fig = p.plot_4dvar('nee', d2, xb=d.xb, xa=xa, awindl=d.len_ren)

import matplotlib.pyplot as plt

plt.show()


Please contact: ewan.pinnington@gmail.com with any queries or issues.
