import numpy as np
import collections as col
import re
import random
import mod_class as mc
import matplotlib.mlab as ml
import datetime as dt
import pickle

class DalecData:
    """
    Data class for the DALEC2 model
    """
    def __init__(self, start_yr, end_yr, ob_str=None, dat_file="../../aliceholtdata/ahdat99_13.csv", k=None):
        """ Extracts data from netcdf file
        :param start_yr: year for model runs to begin as an integer (year)
        :param end_yr: year for model runs to end as an integer (year)
        :param ob_str: string containing observations that will be assimilated (Currently only NEE available)
        :param dat_file: location of csv file to extract data from
        :param k: int to repeat data multiple times
        :return:
        """
        # Extract the data
        data = ml.csv2rec("../../aliceholtdata/ahdat99_13.csv", missing='nan')
        self.flux_data = data[(data['year'] >= start_yr) & (data['year'] < end_yr)]
        self.len_run = len(self.flux_data)
        self.start_yr = start_yr
        self.end_yr = end_yr
        self.time_step = np.arange(self.len_run)
        self.k = k

        # I.C. for carbon pools gCm-2     range
        self.clab = 75.0               # (10,1e3)
        self.cf = 0.0                  # (10,1e3)
        self.cr = 135.0                # (10,1e3)
        self.cw = 14313.0              # (3e3,3e4)
        self.cl = 70.                  # (10,1e3)
        self.cs = 18624.0              # (1e3, 1e5)

        # Parameters for optimization                        range
        self.p1 = 1.1e-5  # theta_min, cl to cs decomp      (1e-5 - 1e-2) day-1
        self.p2 = 0.45  # f_auto, fraction of GPP respired  (0.3 - 0.7)
        self.p3 = 0.01  # f_fol, frac GPP to foliage        (0.01 - 0.5)
        self.p4 = 0.457  # f_roo, frac GPP to fine roots    (0.01 - 0.5)
        self.p5 = 3.  # clspan, leaf lifespan               (1.0001 - 5)
        self.p6 = 4.8e-5  # theta_woo, wood C turnover      (2.5e-5 - 1e-3) day-1
        self.p7 = 6.72e-3  # theta_roo, root C turnover rate(1e-4 - 1e-2) day-1
        self.p8 = 0.024  # theta_lit, litter C turnover     (1e-4 - 1e-2) day-1
        self.p9 = 2.4e-5  # theta_som, SOM C turnover       (1e-7 - 1e-3) day-1
        self.p10 = 0.0193  # Theta, temp dependence exp fact(0.018 - 0.08)
        self.p11 = 90.  # ceff, canopy efficiency param     (10 - 100)
        self.p12 = 140.  # d_onset, clab release date       (1 - 365) (60,150)
        self.p13 = 0.7  # f_lab, frac GPP to clab           (0.01 - 0.5)
        self.p14 = 27.  # cronset, clab release period      (10 - 100)
        self.p15 = 308.  # d_fall, date of leaf fall        (1 - 365) (242,332)
        self.p16 = 35.  # crfall, leaf fall period          (10 - 100)
        self.p17 = 24.2  # clma, leaf mass per area         (10 - 400) g C m-2

        self.param_dict = col.OrderedDict([('theta_min', self.p1),
                                          ('f_auto', self.p2), ('f_fol', self.p3),
                                          ('f_roo', self.p4), ('clspan', self.p5),
                                          ('theta_woo', self.p6), ('theta_roo', self.p7),
                                          ('theta_lit', self.p8), ('theta_som', self.p9),
                                          ('Theta', self.p10), ('ceff', self.p11),
                                          ('d_onset', self.p12), ('f_lab', self.p13),
                                          ('cronset', self.p14), ('d_fall', self.p15),
                                          ('crfall', self.p16), ('clma', self.p17),
                                          ('clab', self.clab), ('cf', self.cf),
                                          ('cr', self.cr), ('cw', self.cw), ('cl', self.cl),
                                          ('cs', self.cs)])
        self.pvals = np.array(self.param_dict.values())

        # Initial guesses to parameter and state variables and their standard deviations
        self.ah_pvals = np.array([9.41e-04, 4.7e-01, 2.8e-01, 2.60e-01, 1.01e+00, 2.6e-04,
                                  2.48e-03, 3.38e-03, 2.6e-06, 1.93e-02, 9.0e+01, 1.4e+02,
                                  4.629e-01, 2.7e+01, 3.08e+02, 3.5e+01, 5.2e+01, 78.,
                                  2., 134., 14257.32, 68.95, 18625.77])

        self.edinburgh_median = np.array([2.29180076e-04,   5.31804031e-01,   6.69448981e-02,
                                          4.46049258e-01,   1.18143120e+00,   5.31584216e-05,
                                          2.25487423e-03,   2.44782152e-03,   7.71092378e-05,
                                          3.82591095e-02,   7.47751776e+01,   1.16238252e+02,
                                          3.26252225e-01,   4.18554035e+01,   2.27257813e+02,
                                          1.20915004e+02,   1.15533213e+02,   1.27804720e+02,
                                          6.02259491e+01,   2.09997016e+02,   4.22672530e+03,
                                          3.67801053e+02,   1.62565304e+03])

        self.edinburgh_mean = np.array([9.80983217e-04,   5.19025559e-01,   1.08612889e-01,
                                        4.84356048e-01,   1.19950434e+00,   1.01336503e-04,
                                        3.22465935e-03,   3.44239452e-03,   1.11320287e-04,
                                        4.14726183e-02,   7.14355778e+01,   1.15778224e+02,
                                        3.20361827e-01,   4.13391057e+01,   2.20529309e+02,
                                        1.16768714e+02,   1.28460812e+02,   1.36541509e+02,
                                        6.86396830e+01,   2.83782534e+02,   6.50600814e+03,
                                        5.98832031e+02,   1.93625350e+03])

        self.edinburgh_std = np.array([2.03001590e-03,   1.16829160e-01,   1.11585876e-01,
                                       2.98860194e-01,   1.16141739e-01,   1.36472702e-04,
                                       2.92998472e-03,   3.11712858e-03,   1.18105073e-04,
                                       1.62308654e-02,   2.04219069e+01,   6.25696097e+00,
                                       1.14535431e-01,   1.40482247e+01,   3.72380005e+01,
                                       2.25938092e+01,   6.41030587e+01,   6.62621885e+01,
                                       3.59002726e+01,   2.19315727e+02,   7.14323513e+03,
                                       5.45013287e+02,   1.27646316e+03])

        self.xa_edc = np.array([1.23000000e-05,   3.13358350e-01,   3.00629189e-01,
                                4.45265166e-01,   1.02310470e+00,   1.22836138e-04,
                                5.04088931e-03,   1.56202990e-03,   1.48252124e-04,
                                7.61636968e-02,   9.27591545e+01,   1.22954168e+02,
                                1.11000000e-02,   4.67979617e+01,   2.87147216e+02,
                                5.51760150e+01,   5.16317404e+01,   1.45000000e+01,
                                1.43000000e+01,   5.01480167e+02,   7.26249788e+03,
                                6.26033838e+02,   2.35514838e+03])

        self.xb = self.edinburgh_median
        # self.B = self.make_b(self.edinburgh_std)
        self.B = pickle.load(open('b_edc.p', 'r'))  # Uses background error cov. matrix B created using ecological
        # dynamical constraints (EDCs) from Bloom and Williams 2015, for more details see Pinningtion et al. 2016

        # Bounds on the parameters for the assimilation
        self.bnds = ((1e-5, 1e-2), (0.3, 0.7), (0.01, 0.5), (0.01, 0.5), (1.0001, 10.),
                     (2.5e-5, 1e-3), (1e-4, 1e-2), (1e-4, 1e-2), (1e-7, 1e-3), (0.018, 0.08),
                     (10, 100), (1, 365), (0.01, 0.5), (10, 100), (1, 365), (10, 100), (10, 400),
                     (10, 1000), (10, 1000), (10, 1000), (100, 1e5), (10, 1000), (100, 2e5))

        self.bnds_tst = ((1e-5, 1e-2), (0.3, 0.7), (0.01, 0.5), (0.01, 0.5), (1.0001, 10.),
                         (2.5e-5, 1e-3), (1e-4, 1e-2), (1e-4, 1e-2), (1e-7, 1e-3), (0.018, 0.08),
                         (10., 100.), (60., 150.), (0.01, 0.5), (10., 100.), (220., 332.), (10., 150.),
                         (10., 400.), (10., 1000.), (10., 1000.), (10., 1000.), (100., 1e5), (10., 1000.),
                         (100., 2e5))

        self.xa = None

        # 'Daily temperatures degC'
        self.t_mean = self.flux_data['t_mean']
        self.t_max = self.flux_data['t_max']
        self.t_min = self.flux_data['t_min']
        self.t_range = np.array(self.t_max) - np.array(self.t_min)

        # 'Driving Data'
        self.I = self.flux_data['i']  # incident radiation
        self.ca = 390.0  # atmospheric carbon
        self.D = self.flux_data['day']  # day of year
        self.year = self.flux_data['year']  # Year
        self.month = self.flux_data['month']  # Month
        self.date = self.flux_data['date']  # Date in month

        datum = dt.datetime(int(self.year[0]), 1, 1)
        delta = dt.timedelta(hours=24)

        # Convert the time values to datetime objects
        self.dates = []
        for t in xrange(len(self.year)):
            self.dates.append(datum + int(t) * delta)

        # Constants for ACM model
        self.acm_williams_xls = np.array([0.0155, 1.526, 324.1, 0.2017,
                                          1.315, 2.595, 0.037, 0.2268,
                                          0.9576])
        self.acm_reflex = np.array([0.0156935, 4.22273, 208.868, 0.0453194,
                                   0.37836, 7.19298, 0.011136, 2.1001,
                                   0.789798])
        self.acm = self.acm_reflex  # (currently using params from REFLEX)
        self.phi_d = -2.5  # max. soil leaf water potential difference
        self.R_tot = 1.  # total plant-soil hydrolic resistance
        self.lat = 0.89133965  # latitude of forest site in radians
        #                        lat = 51.153525 deg, lon = -0.858352 deg

        # misc
        self.ca = 390.0  # atmospheric carbon
        self.radconv = 365.25 / np.pi

        # Background standard deviations for carbon pools & B matrix
        self.sigb_clab = 7.5  # 20%
        self.sigb_cf = 10.0  # 20%
        self.sigb_cw = 1000.  # 20%
        self.sigb_cr = 13.5  # 20%
        self.sigb_cl = 7.0  # 20%
        self.sigb_cs = 1500.  # 20%

        # Observation standard deviations for carbon pools and NEE
        self.sigo_nee = 0.71  # Net ecosystem exchange, g C m-2 day-1
        self.sigo_rtot = 0.71  # Total ecosystem respiration, g C m-2 day-1
        self.sigo_ra = 0.71  # Autotrophic respiration, g C m-2 day-1
        self.sigo_rh = 0.6  # Heterotrohpic respiration, g C m-2 day-1

        self.error_dict = {'nee': self.sigo_nee, 'rtot': self.sigo_rtot, 'rh': self.sigo_rh, 'ra': self.sigo_ra}

        # Extract observations for assimilation
        self.ob_dict, self.ob_err_dict = self.assimilation_obs(ob_str)

    def assimilation_obs(self, ob_str):
        """ Extracts observations and errors for assimilation into dictionaries
        :param obs_str: string of observations separated by commas
        :return: dictionary of observation values, dictionary of corresponding observation errors
        """
        possible_obs = ['nee', 'rtot', 'rh', 'ra']
        obs_lst = re.findall(r'[^,;\s]+', ob_str)
        obs_dict = {}
        obs_err_dict = {}
        for ob in obs_lst:
            if ob not in possible_obs:
                raise Exception('Invalid observations entered, please check \
                                 function input')
            else:
                obs = self.flux_data[ob]
                obs_dict[ob] = obs
                obs_err_dict[ob] = (obs/obs) * self.error_dict[ob]

        return obs_dict, obs_err_dict

    @staticmethod
    def make_b(b_std):
        """ Creates diagonal B matrix.
        :param b_std: array of standard deviations corresponding to each model parameter
        :return: 23 x 23 diagonal background error covariance matrix
        """
        b_mat = b_std**2 * np.eye(23)
        return b_mat


class DalecDataTwin(DalecData):
    """
    Dalec twin data class, needs more work before use! Particularly on the creation of synthetic observations.
    """
    def __init__(self, start_date, end_date, ob_str, dat_file="../../aliceholtdata/ahdat99_13.csv"):
        DalecData.__init__(self, start_date, end_date, ob_str, dat_file)

        # self.d = DalecData(start_date, end_date, ob_str, nc_file)
        self.m = mc.DalecModel(self)

        # Define truth and background
        self.x_truth = self.edinburgh_median
        self.st_dev = 0.10*self.x_truth
        self.B = self.make_b(self.st_dev)
        # self.xb = self.random_pert(self.random_pert(self.x_truth))
        self.xb = np.array([2.53533992e-04,   5.85073161e-01,   7.43127332e-02,
                            4.99707798e-01,   1.38993876e+00,   6.11913792e-05,
                            2.58484324e-03,   2.79379720e-03,   8.72422101e-05,
                            4.35144260e-02,   8.73669864e+01,   1.29813051e+02,
                            3.87867223e-01,   4.69894281e+01,   2.78080852e+02,
                            9.15080347e+01,   1.36269157e+02,   1.44176657e+02,
                            6.71153814e+01,   2.42199267e+02,   4.96249386e+03,
                            4.15128028e+02,   1.90797697e+03])
        # Extract observations for assimilation
        self.ob_dict, self.ob_err_dict = self.create_twin_data(ob_str)

    def create_twin_data(self, ob_str, err_scale=0.25):
        """ Creates a set of twin modelled observations corresponding to the same positions as the true observations
        :param ob_str: str of observations
        :param err_scale: factor by which to scale observation error and added gaussian noise
        :return: observation dictionary, observation error dictionary
        """
        possible_obs = ['nee', 'rtot', 'rh', 'ra']
        obs_lst = re.findall(r'[^,;\s]+', ob_str)
        obs_dict = {}
        obs_err_dict = {}
        mod_lst = self.m.mod_list(self.x_truth)
        for ob in obs_lst:
            if ob not in possible_obs:
                raise Exception('Invalid observations entered, please check \
                                 function input')
            else:
                obs = self.ob_dict[ob]  # actual observations
                mod_obs = (obs/obs) * self.m.oblist(ob, mod_lst)  # modelled observation corresponding to same
                # position as actual obs
                # adding error to modelled observations
                mod_ob_assim = np.array([mod_ob + random.gauss(0, err_scale*self.error_dict[ob])
                                         for mod_ob in mod_obs])
                obs_dict[ob] = mod_ob_assim
                obs_err_dict[ob] = err_scale*self.ob_err_dict[ob]
        return obs_dict, obs_err_dict

    def random_pert(self, pvals):
        """ Perturbs parameter values with given standard deviation
        :param pvals: parameter values to perturb
        :return: perturbed parameters
        """
        pval_approx = np.ones(23)*-9999.
        x = 0
        for p in pvals:
            pval_approx[x] = p + random.gauss(0, self.st_dev[x])
            if self.bnds[x][1] < pval_approx[x]:
                pval_approx[x] = self.bnds[x][1] - abs(random.gauss(0, self.bnds[x][1]*0.001))
            elif self.bnds[x][0] > pval_approx[x]:
                pval_approx[x] = self.bnds[x][0] + abs(random.gauss(0, self.bnds[x][0]*0.001))

            x += 1

        return pval_approx

    def random_pert_uniform(self, pvals):
        """ Perturbs parameter values with given standard deviation
        :param pvals: parameter values to perturb
        :return: perturbed parameters
        """
        pval_approx = np.ones(23)*-9999.
        xt = self.x_truth
        x = 0
        for p in pvals:
            pval_approx[x] = p + random.uniform(-0.1*xt[x], 0.1*xt[x])
            if 0.3 < abs(pval_approx[x] - self.x_truth[x])/self.x_truth[x]:
                while 0.3 < abs(pval_approx[x] - self.x_truth[x])/self.x_truth[x]:
                    pval_approx[x] = pval_approx[x] - abs(random.uniform(-0.1*xt[x], 0.1*xt[x]))
            if abs(pval_approx[x] - self.x_truth[x])/self.x_truth[x] < 0.12:
                while abs(pval_approx[x] - self.x_truth[x])/self.x_truth[x] < 0.12:
                    pval_approx[x] = pval_approx[x] + abs(random.uniform(-0.1*xt[x], 0.1*xt[x]))
            if self.bnds[x][1] < pval_approx[x]:
                pval_approx[x] = self.bnds[x][1] - abs(random.gauss(0, self.bnds[x][1]*0.001))
            elif self.bnds[x][0] > pval_approx[x]:
                pval_approx[x] = self.bnds[x][0] + abs(random.gauss(0, self.bnds[x][0]*0.001))
            x += 1
        return pval_approx

    def test_pvals(self, pvals):
        """ Test if a parameter set falls within the bounds or not
        :param pvals: parameter values to test
        :return:
        """
        x = 0
        for bnd in self.bnds:
            if bnd[0] < pvals[x] < bnd[1]:
                print '%x in bnds' %x
            else:
                print '%x not in bnds' %x
            x += 1
        return pvals


