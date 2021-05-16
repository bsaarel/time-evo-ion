from qutip import *
from qutip.ui.progressbar import TextProgressBar
# import numba, sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import eval_genlaguerre
import csv
import lmfit
import scipy.optimize as op
import sys
import pickle

# the mock-0.3.1 dir contains testcase.py, testutils.py & mock.py
sys.path.append("/Users/Ben/Library/Mobile Documents/com~apple~CloudDocs/Documents/GitHub/sqip_data_analysis/rabi_to_heating_rate_python3/heating_rate_analysis")
from sqip_rate_from_flops import *
from sqip_rabi_flop_fitter import *


def load_files(path, filename):
    with open(path + filename, 'rb') as data_file:
        return pickle.load(data_file)
class rabiObj():
    def __init__(self,rho0,coupling,tlist,tlist_rabi,N=100,color='black',label = '',alpha = 0,nbpts=10):
        self.rho0 = rho0
        self.rhoTE = rho0
        self.rhof = rho0
        self.cp = coupling
        self.N = N
        self.a = qutip.destroy(N)
        self.H = self.a.dag() * self.a
        self.rabi_fit = False
        self.rabi_flop_dict = {}
        self.tlist_rabi = tlist_rabi
        self.tlist = tlist
        self.runfree = True
        self.nbpts = nbpts
        self.rabi_0 = 0.15
        self.nbarlist = []
        self.nbarerrs = []
        self.trap_freq= 1e6
        self.theta = 0
        self.color = color
        self.label = label
        self.alpha = alpha

    def time_evo(self):
        print("calculating time evolution...")
        kappa = self.cp # coupling to oscillator
        # collapse operators
        c_op_list = []
        n_th_a = 6*10**6 # temperature with average of 2 excitations
        rate = kappa * (1 + n_th_a)
        if rate > 0.0:
            c_op_list.append(np.sqrt(rate) * self.a) # decay operators
        rate = kappa * n_th_a
        if rate > 0.0:
            c_op_list.append(np.sqrt(rate) * self.a.dag()) # excitation operators
        # find steady-state solution
        final_state = qutip.steadystate(self.H, c_op_list)
        # find expectation value for particle number in steady state
        # fexpt = qutip.expect(a.dag() * a, final_state)
        if self.rabi_fit == True:
            self.rhoTE = qutip.mesolve(self.H, self.rho0, self.tlist, c_op_list,progress_bar=TextProgressBar(self.N))
            self.sample_waits(self.nbpts)
            self.save_files("/Users/Ben/Library/Mobile Documents/com~apple~CloudDocs/Documents/GitHub/ionLifetimes/time-evo-ion/N400NbTest/",self.label)
        else:
            return qutip.mesolve(self.H, self.rho0, self.tlist, c_op_list, self.a.dag() * self.a, progress_bar=TextProgressBar(N))

    def compute_rabi_flop(self,index):
        rabi_flop = np.zeros(len(self.tlist_rabi))
        etas = [calc_eta(self.trap_freq, self.theta)]
        eta = etas[0]
        rabi_0 = self.rabi_0
        accum = 0
        for x in range(self.N):
            if index == 0:
               pn = self.rho0.diag()[x]
            else:
                pn = self.rhoTE.states[index].diag()[x]
            accum = accum+pn
            rabi_freq = rabi_0*np.exp(-eta**2/2)*eval_genlaguerre(x,0,eta**2)
            rabi_flop = rabi_flop + pn*(np.sin(self.tlist_rabi*rabi_freq))**2
        return rabi_flop

    def sample_waits(self,pts,color="black"):
        tfactor = self.tlist[len(self.tlist)-1]/pts
        ifactor = len(self.tlist)/pts
        self.rabi_flop_dict = {}
        for idx in range(pts):
            flop = self.compute_rabi_flop(int(ifactor*idx))
            self.rabi_flop_dict[int(tfactor*idx)] = flop
            # plt.plot(self.tlist_rabi,flop,label=idx*factor,color=color)

    def save_files(self,path,filename,add_on = ""):
        config_dictionary = {'remote_hostname': 'google.com', 'remote_port': 80}
        # with open('config.dictionary', 'wb') as config_dictionary_file:
        with open(path + filename + add_on, mode='wb') as data_file:
            # Step 3
            pickle.dump(self, data_file)
        # for waittime in self.rabi_flop_dict.keys():
        #     with open(path+filename+str(waittime)+add_on+".csv", mode='w') as data_file:
        #         data_writer = csv.writer(data_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        #         for i in range(len(self.tlist_rabi)):
        #             data_writer.writerow([self.tlist_rabi[i],self.rabi_flop_dict[waittime]])
    def generate_heating_rates(self):
        print("sampling time-evolved state")
        nbarlist = []
        nbarerrs = []
        pitimes = []
        pitimeserr = []
        deltas = []
        deltaserr = []
        excitation_scaling_list = []
        excitation_scaling_errs = []
        times = np.sort(list(self.rabi_flop_dict.keys()))
        etas = [calc_eta(self.trap_freq, self.theta)]
        eta = etas[0]
        # f1 = plt.figure()
        # ax = f1.add_subplot()
        for key in times:
            # importf data and calculate errors based on 100 experiments
            t = self.tlist_rabi
            p = self.rabi_flop_dict[key]
            perr = [np.sqrt(x * (1 - x) / 100.) for x in p]
            for index in range(0, len(p)):
                if perr[index] == 0:
                    perr[index] = np.sqrt(0.01 * (1 - 0.01) / 100.)

            # model for Rabi flops, use 0 for sidebands, +-1 etc for sidebands..
            te = rabi_flop_time_evolution(0, eta,
                                          nmax=int(1e4))  # if you get the error that the hilbert space is too small, then increase nmax
            params = lmfit.Parameters()
            params.add('delta', value=0.0, vary=False)
            # params.add('delta',value = 0.0 ,vary=True,min=0.01,max=0.04)
            params.add('nbar', value=30, vary=True,min=0.01, max=200.0)
            excitation_scaling = 1 # guess how well OP works
            time_2pi = np.pi/self.rabi_0
            # if key == 0 or self.runfree:
            #      print("boom!")
            #      params.add('time_2pi', value=time_2pi, vary=True)
            # else:
            params.add('time_2pi', value=time_2pi, vary=False)

            params.add('excitation_scaling', value=excitation_scaling, vary=False, min=0.95, max=1.0)

            ##                params.add('nbar1',value = nbar,min=0.0,max=100.0,vary=True)
            ##                params.add('nbar2',value = nbar,min=0.0,max=100.0,vary=True)
            ##
            guess_values = te.compute_evolution_thermal(params['nbar'].value, params['delta'].value,
                                                        params['time_2pi'].value, t, params['excitation_scaling'].value)

            def rabi_fit_thermal(params, t, data, err):
                model = te.compute_evolution_thermal(params['nbar'].value, params['delta'].value,
                                                     params['time_2pi'].value, t, params['excitation_scaling'].value)
                resid = model - data
                weighted = [np.sqrt(resid[x] ** 2/ err[x] ** 2) for x in range(len(err))]
                # plt.plot(resid)
                # plt.show()
                # sum_weighted = sum(weighted)
                # print "sum of residuals: " + str(sum_weighted)
                return weighted

            result = lmfit.minimize(rabi_fit_thermal, params, method='leastsq', max_nfev=10**9, xtol=1.e-8, ftol=1.e-8,
                                    args=(t, np.real(p), np.real(perr)))
            # plt.plot(t,np.real(p))
            # result = lmfit.minimize(rabi_fit_thermal,params, args = (t,np.real(p),np.real(perr)))
            print("reduced chi^2 " + str(result.redchi))
            params = result.params
            if key == 0:
                time_2pi = params['time_2pi'].value
                excitation_scaling = params['excitation_scaling'].value
            tvec = np.linspace(0, 50, 50)
            fit_values = te.compute_evolution_thermal(params['nbar'].value, params['delta'].value,
                                                      params['time_2pi'].value, tvec,
                                                      params['excitation_scaling'].value)
            nbarlist = np.append(nbarlist, params['nbar'].value)
            if params['nbar'].stderr== None:
                params['nbar'].stderr =0
            nbarerrs = np.append(nbarerrs, params['nbar'].stderr)
            print(params['time_2pi'].stderr)

            if params['time_2pi'].stderr == None:
                params['time_2pi'].stderr = 0
            print(str(key) + "   " + str(params['time_2pi'].value) + " +- " + str(params['time_2pi'].stderr))
            print(params['time_2pi'])
            pitimes = np.append(pitimes, params['time_2pi'].value / 2.0)
            # pitimeserr = np.append(pitimeserr,params['time_2pi'].stderr/2.0)
            deltas = np.append(deltas, params['delta'].value)
            deltaserr = np.append(deltaserr, params['delta'].stderr)
            excitation_scaling_list = np.append(excitation_scaling_list, params['excitation_scaling'].value)
            excitation_scaling_errs = np.append(excitation_scaling_errs, params['excitation_scaling'].stderr)
            # print 'optical pumping estimate from fit:' + str(params['excitation_scaling'].value)
            print('ok')
            # pyplot.plot(list(np.arange(np.alen(out))), out)
            # plt.plot(t, p, 'r', label='data' + str(self.alpha), marker='^',linestyle='none',color = self.color)
            # plt.plot(tvec, fit_values, 'r', label='fitted ' + str(self.alpha),linestyle = "--",color = self.color)
            # plt.title('flops for trap freq ' + str(int(self.trap_freq / 1000000)) + ' MHz')
            # # pyplot.plot(t,guess_values,'r',label = 'guess '+ str(key),color='black')
            # # ax.title('flops for trap freq ' + str(trap_freq / 1000000) + ' MHz')
            # plt.xlabel('Time (\u03BCs)')
            # plt.ylabel('P1(t)')
            # plt.legend()
            # plt.show()
        self.nbarlist = nbarlist
        self.nbarerrs = nbarerrs

        return (times, nbarlist, nbarerrs, excitation_scaling_list, excitation_scaling_errs)
    def calculate_rate(self,rabi_data,color='black',label = '',style = 'solid'):

        # linear model for heating rate
        def f(x, rate, offset):
            return rate * x + offset

        #fit using the nbar from the thermal dist fits to the numerically generated rabi flops
        if rabi_data:
            # getting the times we sampled nbar for
            vals = list(self.rabi_flop_dict.keys())
            # do least squares fit of the linear model to extract the heating rate
            popt, pcov = op.curve_fit(f, vals, self.nbarlist, p0=[1, 20], sigma=self.nbarerrs, absolute_sigma=True)
            # getting the standard error associated with the heating rate linear regression fit
            perr = np.sqrt(np.diag(pcov))
            perr = perr[0]
            # getting the heating rate
            rate = popt[0]
            er1 = plt.errorbar(self.rabi_flop_dict.keys(), self.nbarlist, self.nbarerrs, color=color, marker='^',
                               linestyle="none", label= "thermal fit,"+label + ": $d\overline{n}$/dt =" + str(
                    round(rate, 2)) + "+-" + str(round(perr, 2)))

        #fit using the nbar directly from the density matrix evaluated at each time step
        else:
            nblist = []
            for rho in self.rhoTE.states:
                expt = expect(self.H, rho)
                nblist.append(expt)
            popt, pcov = op.curve_fit(f, self.tlist, np.real(nblist), p0=[1, 20], absolute_sigma=True)
            # getting the standard error associated with the heating rate linear regression fit
            perr = np.sqrt(np.diag(pcov))
            perr = perr[0]
            # getting the heating rate
            rate = popt[0]
            pt1 = plt.plot(self.tlist, nblist, color=color, linestyle=style,
                           label='$Actual Value$,' +label+ str(round(rate, 2)) + "+-" + str(round(perr, 2)))
        # plt.show()
        return rate,perr